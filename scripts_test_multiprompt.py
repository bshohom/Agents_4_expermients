"""Run legacy prompt-based proposer testing across multiple prompt files.

This keeps the existing prompt-testing workflow available while the main
`orchestrator.py` defaults to the Scripts multi-agent loop.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from agents.llm_client import LocalLLMClient
from agents.rag_cards import PaperCardRAG
from agents.proposer import ProposerAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--bom_file", required=True)
    parser.add_argument("--prompt_glob", default="prompts/*.txt")
    parser.add_argument(
        "--output_dir",
        default="outputs/multiprompt",
        help="Directory where per-prompt outputs and summary are saved.",
    )
    parser.add_argument(
        "--user_goal",
        default="Find one feasible literature-grounded experiment for the available BOM.",
    )
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--skip_llm", action="store_true")
    return parser.parse_args()


def build_retrieval_query(bom: dict, user_goal: str) -> str:
    equipment = ", ".join(bom.get("available_equipment", [])[:10])
    consumables = ", ".join(bom.get("available_consumables", [])[:10])
    constraints = json.dumps(bom.get("goal_constraints", {}), sort_keys=True)
    return (
        f"{user_goal}\n"
        f"Available equipment: {equipment}\n"
        f"Available consumables: {consumables}\n"
        f"Constraints: {constraints}"
    )


def main() -> None:
    args = parse_args()

    with open(args.bom_file, "r", encoding="utf-8") as f:
        bom = json.load(f)

    prompt_paths = sorted(Path().glob(args.prompt_glob))
    if not prompt_paths:
        raise SystemExit(f"No prompt files matched --prompt_glob={args.prompt_glob}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rag = PaperCardRAG(index_dir=args.index_dir, top_k=args.top_k)
    llm_client = LocalLLMClient()

    retrieval_query = build_retrieval_query(bom=bom, user_goal=args.user_goal)
    retrieved_cards = rag.retrieve(retrieval_query, top_k=args.top_k)

    run_timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    per_prompt_results: list[dict] = []

    for prompt_path in prompt_paths:
        proposer = ProposerAgent(prompt_file=prompt_path, llm_client=llm_client)

        response = None
        if not args.skip_llm:
            response = proposer.run(
                bom=bom,
                user_goal=args.user_goal,
                retrieved_cards=retrieved_cards,
            )

        result = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "prompt_file": str(prompt_path),
            "user_goal": args.user_goal,
            "retrieval_query": retrieval_query,
            "retrieved_cards": retrieved_cards,
            "response": response,
        }

        prompt_stem = prompt_path.stem
        output_file = run_dir / f"{prompt_stem}.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        per_prompt_results.append(
            {
                "prompt_file": str(prompt_path),
                "output_file": str(output_file),
                "response_chars": len(response or ""),
            }
        )

        print(f"Saved: {output_file}")

    summary = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "mode": "legacy-multiprompt",
        "index_dir": args.index_dir,
        "bom_file": args.bom_file,
        "prompt_glob": args.prompt_glob,
        "skip_llm": args.skip_llm,
        "run_dir": str(run_dir),
        "prompt_count": len(prompt_paths),
        "results": per_prompt_results,
    }

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
