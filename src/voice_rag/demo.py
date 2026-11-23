"""Offline text-only demo of conversational query expansion and RAG, no API calls."""

from __future__ import annotations

import argparse
from pathlib import Path

from voice_rag.conversation import ConversationQuery
from voice_rag.retrieval import Index, format_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/knowledge-index.json")
    args = parser.parse_args()
    index = Index.load(Path(args.index))
    memory = ConversationQuery()
    print("Offline RAG inspection mode. Type /quit to exit (no LLM voice answer).")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question in {"/quit", "/exit"}:
            break
        query, needs_retrieval = memory.prepare(question)
        print(f"Retrieval query: {query}")
        print(format_context(index.search(query)) if needs_retrieval else "Social turn; RAG skipped")


if __name__ == "__main__":
    main()
