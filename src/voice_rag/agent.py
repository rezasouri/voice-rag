"""LiveKit conversation-first Persian agent with guaranteed per-turn retrieval."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    TurnHandlingOptions,
    cli,
)
from livekit.plugins import openai, silero

from voice_rag.conversation import ConversationQuery, SYSTEM_PROMPT
from voice_rag.ingest import OpenAIEmbedder
from voice_rag.retrieval import Index, format_context

load_dotenv()
load_dotenv(".env.local", override=True)


class ConversationalVoiceRAG(Agent):
    """Uses previous conversation turns and pulls fresh knowledge on *every factual turn*."""

    def __init__(self, index: Index, *, semantic: bool):
        super().__init__(instructions=SYSTEM_PROMPT)
        self._index = index
        self._queries = ConversationQuery()
        self._embedder = OpenAIEmbedder() if semantic else None

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        query, should_search = self._queries.prepare(new_message.text_content or "")
        if not should_search:
            return
        vector: list[float] | None = None
        if self._embedder:
            try:
                vector = (await asyncio.to_thread(self._embedder.embed, [query]))[0]
            except Exception:
                # Transient embedding failures do not block spoken replies; BM25 still works.
                vector = None
        hits = self._index.search(query, k=4, query_embedding=vector)
        turn_ctx.add_message(
            role="assistant",
            content="Context for the current user's question (not a spoken assistant answer).\n"
            + format_context(hits),
        )


server = AgentServer()


@server.rtc_session(agent_name="conversational-voice-rag")
async def agent_entrypoint(ctx: JobContext) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in .env or .env.local to run the voice agent")
    index_path = Path(os.getenv("VOICE_RAG_INDEX", "data/knowledge-index.json"))
    if not index_path.exists():
        raise FileNotFoundError(f"Build the knowledge index first: {index_path}")
    index = await asyncio.to_thread(Index.load, index_path)
    semantic_mode = os.getenv("VOICE_RAG_SEMANTIC_QUERY", "auto").lower()
    has_embeddings = bool(index.passages) and all(p.embedding for p in index.passages)
    semantic = has_embeddings and semantic_mode != "off"

    session = AgentSession(
        stt=openai.STT(
            model=os.getenv("VOICE_RAG_STT_MODEL", "gpt-4o-mini-transcribe"),
            language=os.getenv("VOICE_RAG_LANGUAGE", "fa"),
        ),
        llm=openai.responses.LLM(model=os.getenv("VOICE_RAG_LLM_MODEL", "gpt-4.1-mini")),
        tts=openai.TTS(
            model=os.getenv("VOICE_RAG_TTS_MODEL", "gpt-4o-mini-tts"),
            voice=os.getenv("VOICE_RAG_TTS_VOICE", "ash"),
            instructions="Speak in natural conversational Persian, with short, warm sentences and clear pronunciation.",
        ),
        vad=silero.VAD.load(),
        # The LiveKit semantic turn detector does not currently include Persian.
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            endpointing={"mode": "fixed", "min_delay": 0.8, "max_delay": 2.0},
            interruption={"enabled": True, "mode": "vad"},
        ),
    )
    await session.start(room=ctx.room, agent=ConversationalVoiceRAG(index, semantic=semantic))
    await session.generate_reply(
        instructions="Say one brief friendly greeting in Persian, and invite the user to talk freely."
    )


def main() -> None:
    cli.run_app(server)


if __name__ == "__main__":
    main()
