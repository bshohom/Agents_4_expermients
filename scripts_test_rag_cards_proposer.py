"""Standalone test for rag_cards + proposer.

This is the concrete test path requested by the user:
- load the persisted card index
- run retrieval for a BOM-aware query
- load a prompt file
- call the proposer against the local LLM server
- print and save output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.llm_client import LocalLLMClient
from agents.rag_cards import PaperCardRAG
from agents.proposer import ProposerAgent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--index_dir", required=True)
    p.add_argument("--prompt_file", required=True)
    p.add_argument("--bom_file", required=True)
    p.add_argument("--user_goal", default="Find one feasible literature-grounded experiment for the available BOM.")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--output_file", default="outputs/test_rag_cards_proposer_output.json")
    p.add_argument("--skip_llm", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.bom_file, "r", encoding="utf-8") as f:
        bom = json.load(f)

    rag = PaperCardRAG(index_dir=args.index_dir, top_k=args.top_k)
    client = LocalLLMClient()
    proposer = ProposerAgent(prompt_file=args.prompt_file, llm_client=client)

    query = (
        f"{args.user_goal}\n"
        f"Available equipment: {', '.join(bom.get('available_equipment', []))}\n"
        f"Available consumables: {', '.join(bom.get('available_consumables', []))}"
    )

    cards = rag.retrieve(query, top_k=args.top_k)
    response = None
    if not args.skip_llm:
        response = proposer.run(
            bom=bom,
            user_goal=args.user_goal,
            retrieved_cards=cards,
        )

    result = {
        "query": query,
        "retrieved_cards": cards,
        "response": response,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=== Retrieved cards ===")
    print(json.dumps(cards, indent=2))

    if response is not None:
        print("\n=== Model response ===")
        print(response)
    else:
        print("\nSkipped LLM call; retrieval-only mode succeeded.")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=== Retrieved cards ===")
    print(json.dumps(cards, indent=2))
    print("\n=== Model response ===")
    print(response)
    print(f"\nSaved standalone test output to: {out_path}")


if __name__ == "__main__":
    main()
