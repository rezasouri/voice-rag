"""Build the local RAG index from a knowledge folder."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from voice_rag.retrieval import Index


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI

        self._client = OpenAI()
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [e.embedding for e in sorted(response.data, key=lambda item: item.index)]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Create or update Voice RAG knowledge index")
    parser.add_argument("--source", type=Path, default=Path("knowledge"))
    parser.add_argument("--output", type=Path, default=Path("data/knowledge-index.json"))
    parser.add_argument("--semantic", action="store_true", help="Use OpenAI embeddings (paid API)")
    args = parser.parse_args()
    if args.semantic and not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY must be set to build a semantic index")
    idx = Index.build(args.source, embedder=OpenAIEmbedder() if args.semantic else None)
    idx.save(args.output)
    print(f"Indexed {len(idx.passages)} passage(s) -> {args.output}")
    if not idx.passages:
        print("WARNING: The knowledge directory yielded no passages")


if __name__ == "__main__":
    main()
