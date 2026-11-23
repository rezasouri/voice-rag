# Conversational Voice RAG — Persian-first

Independent prototype for **natural spoken conversation**, with RAG as a grounding capability rather than the entire conversational flow.

## What it does

- **Real voice interaction** through microphone and speaker in LiveKit console mode.
- **Interruption / barge-in**: user speech can interrupt ongoing assistant audio; tuned with Silero VAD.
- **Multi-turn memory** in each LiveKit AgentSession, with pronoun/follow-up expansion for search queries.
- **Turn-by-turn retrieval**: every non-social voice turn retrieves from indexed documents (not left to optional LLM tool-calling).
- **Citations**: passages carry source filenames (PDF pages if present), which the assistant can cite verbally.
- **RAG** with offline BM25 retrieval; optional semantic embeddings + reciprocal-rank fusion. Real knowledge is provided by the user.
- **Persian** transcription and spoken responses; models can be changed by configuration.
- **Private-by-default source storage**: index stays local. However, **audio, transcriptions, spoken answers, retrieved excerpts and (when enabled) embedding requests reach the configured OpenAI API**; this is not an offline/private inference stack.

## Requirements

- Python 3.11–3.13, `uv`, working microphone and speaker.
- An **OpenAI API key** for STT, LLM and TTS. An API bill is possible.
- No LiveKit account is required for **console mode** (the initial local interactive experience).
- For browser/mobile/telephone LiveKit rooms, configure LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET and run dev/start. A web frontend is **not included** in this package.

## Run it locally

```bash
uv sync --group dev --extra documents
cp .env.example .env
# Open .env and set OPENAI_API_KEY; don't commit it

# Build the knowledge index from files in ./knowledge
uv run voice-rag-index --source knowledge --output data/knowledge-index.json

# Optional semantic retrieval, sends document chunks to OpenAI for embeddings:
uv run voice-rag-index --source knowledge --output data/knowledge-index.json --semantic

# Download local Silero speech detection model weights:
uv run python src/agent.py download-files

# Talk to it from your terminal microphone / speakers (local, no LiveKit account):
uv run python src/agent.py console

# Inspect retrieval and conversational follow-ups without external API requests:
uv run python -m voice_rag.demo

# Tests (no API calls needed):
uv run pytest -q
```

Tip: use headphones initially to avoid loudspeaker-to-microphone echo causing false interruptions. Console mode also supports text testing.

## Add knowledge

Place UTF-8 `.txt` or `.md`, PDF with extractable text, or `.docx` under `knowledge/`. Re-run `voice-rag-index`. PDFs consisting of scans need an OCR ingestion adapter (not included). Sample knowledge is explicitly fictional, intended only for demonstration.

## How the conversation works

```text
Microphone / LiveKit room
        | streaming audio
Silero VAD -> Persian STT -> user utterance
                                 |
                        conversation state
                                 |
                  context-aware RAG query
                                 |
                  BM25 + optional embeddings
                                 |
                  snippets with source IDs
                                 |
                   conversational LLM
                                 |
               streaming TTS -> speaker
                        ^
           barge-in interrupt loop
```

Design choice: We use **VAD-only endpointing** instead of LiveKit's semantic turn detector, which currently does not list Persian among its supported languages. Fixed 0.8-second endpointing is a starting point, not a validated optimum. Tune using real Persian dialogues and pauses.

## Conversational acceptance scenarios

See [`docs/acceptance.md`](docs/acceptance.md) for hands-on tests focused on naturalness, memory, and interruption handling.

## Known limits (MVP, not production)

- Not end-to-end voice tested in this artifact build (requires credentials, microphone and audio device).
- Per-room conversation memory only; no database-backed long-term memory or user identity/ACL.
- No browser UI or SIP integration bundled yet; LiveKit-compatible deployment extension possible.
- Streaming audio is managed by LiveKit, but STT endpointing and TTS latency vary by provider and network.
- Persian pronunciation, latency, echo, interruptions and noisy environments still require on-device evaluation.
- BM25 is lexical, and optional embeddings improve semantic retrieval. Retrieval confidence cutoffs need evaluation data.
- No streaming audio capture/recording by the app; review data retention/security of chosen external providers separately.

## References

- https://docs.livekit.io/agents/logic/turns/
- https://docs.livekit.io/agents/logic/turns/tuning/
- https://docs.livekit.io/agents/logic/external-data/
- https://docs.livekit.io/agents/models/stt/openai/
- https://docs.livekit.io/agents/models/tts/openai/
- https://docs.livekit.io/agents/server/startup-modes/
