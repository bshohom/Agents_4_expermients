from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_MAX_ROUNDS = 5
PROMPT_SPLIT_MARKER = "\n<<<PROMPT>>>\n"


def should_stop(advice: dict, round_idx: int) -> bool:
    status = advice.get("bom_check", {}).get("status", "").strip()

    if advice.get("parse_failed", False) and round_idx >= 1:
        return True

    if status == "strongly_feasible" and round_idx >= 1:
        return True

    if status == "mostly_feasible" and round_idx >= 2:
        return True

    return False


def print_round_summary(round_idx: int, rag1_output: dict, rag2_advice: dict) -> None:
    proposal = rag1_output.get("rag1_proposal", "")
    status = rag2_advice.get("bom_check", {}).get("status", "unknown")
    reason = rag2_advice.get("bom_check", {}).get("reason", "")

    print("\n" + "=" * 100)
    print(f"ROUND {round_idx + 1} SUMMARY")
    print("=" * 100)
    print(f"RAG2 status: {status}")
    print(f"Reason: {reason}")

    msg = rag2_advice.get("message_to_rag1", "")
    if msg:
        print(f"\nMessage to RAG1:\n{msg}")

    narrowing = rag2_advice.get("narrowing_advice", [])
    if narrowing:
        print("\nTop narrowing advice:")
        for i, item in enumerate(narrowing[:3], 1):
            advice_text = item.get("advice", "")
            why_text = item.get("why", "")
            print(f"{i}. {advice_text}")
            if why_text:
                print(f"   why: {why_text}")

    if proposal:
        print("\nLatest proposal preview:")
        print(proposal[:1500])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-agent RAG proposal loop.")
    parser.add_argument("--experiment-index-dir", type=Path, default=Path("outputs/paper_memory_storage_experiment"))
    parser.add_argument("--science-index-dir", type=Path, default=Path("outputs/paper_memory_storage_science"))
    parser.add_argument("--chunks-labeled-path", type=Path, default=Path("outputs/chunks_labeled.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/orchestrator_runs"))
    parser.add_argument("--proposal-prompts-file", type=Path, default=None)
    parser.add_argument("--reviewer-prompts-file", type=Path, default=None)
    parser.add_argument("--bom-file", type=Path, default=None, help="Optional JSONL file of BOM variants.")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--embed-model", type=str, default=None)
    parser.add_argument("--proposal-temperatures", type=str, default="0.0")
    parser.add_argument("--reviewer-temperature", type=float, default=0.0)
    parser.add_argument("--seed-values", type=str, default="0")
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def parse_float_list(csv_text: str) -> list[float]:
    values = []
    for chunk in csv_text.split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(float(chunk))
    return values or [0.0]


def parse_int_list(csv_text: str) -> list[int]:
    values = []
    for chunk in csv_text.split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(int(chunk))
    return values or [0]


def load_bom_variants(path: Path | None) -> list[dict]:
    if path is None:
        return [{"name": "default_bom", "bom": None}]

    variants = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            variants.append(
                {
                    "name": row["name"],
                    "bom": row["bom"],
                }
            )

    if not variants:
        return [{"name": "default_bom", "bom": None}]
    return variants


def load_prompt_variants(path: Path | None, label: str) -> list[dict]:
    if path is None:
        return [{"name": f"default_{label}", "text": ""}]

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return [{"name": f"default_{label}", "text": ""}]

    prompts: list[dict] = []

    if path.suffix.lower() == ".jsonl":
        for idx, line in enumerate(raw.splitlines()):
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, str):
                prompts.append({"name": f"{label}_{idx}", "text": item})
            else:
                prompts.append(
                    {
                        "name": item.get("name", f"{label}_{idx}"),
                        "text": item.get("prompt", item.get("text", "")),
                    }
                )
        return prompts or [{"name": f"default_{label}", "text": ""}]

    if path.suffix.lower() == ".json":
        item = json.loads(raw)
        if isinstance(item, list):
            for idx, entry in enumerate(item):
                if isinstance(entry, str):
                    prompts.append({"name": f"{label}_{idx}", "text": entry})
                else:
                    prompts.append(
                        {
                            "name": entry.get("name", f"{label}_{idx}"),
                            "text": entry.get("prompt", entry.get("text", "")),
                        }
                    )
            return prompts or [{"name": f"default_{label}", "text": ""}]

    if PROMPT_SPLIT_MARKER in raw:
        for idx, block in enumerate(raw.split(PROMPT_SPLIT_MARKER)):
            block = block.strip()
            if block:
                prompts.append({"name": f"{label}_{idx}", "text": block})
    else:
        for idx, line in enumerate(raw.splitlines()):
            line = line.strip()
            if line:
                prompts.append({"name": f"{label}_{idx}", "text": line})

    return prompts or [{"name": f"default_{label}", "text": ""}]


def resolve_task_id(explicit_task_id: int | None) -> int:
    if explicit_task_id is not None:
        return explicit_task_id

    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_task_id is not None:
        return int(slurm_task_id)

    return 0


def select_combo(
    task_id: int,
    proposal_prompts: list[dict],
    reviewer_prompts: list[dict],
    bom_variants: list[dict],
    proposal_temperatures: list[float],
    seed_values: list[int],
) -> dict:
    n_proposal = len(proposal_prompts)
    n_reviewer = len(reviewer_prompts)
    n_bom = len(bom_variants)
    n_temp = len(proposal_temperatures)
    n_seed = len(seed_values)

    total = n_proposal * n_reviewer * n_bom * n_temp * n_seed
    if task_id < 0 or task_id >= total:
        raise ValueError(
            f"task_id={task_id} is out of range for "
            f"{n_proposal} proposer x {n_reviewer} reviewer x {n_bom} bom x "
            f"{n_temp} proposal_temp x {n_seed} seed = {total} combinations."
        )

    proposal_idx = task_id // (n_reviewer * n_bom * n_temp * n_seed)
    remainder = task_id % (n_reviewer * n_bom * n_temp * n_seed)

    reviewer_idx = remainder // (n_bom * n_temp * n_seed)
    remainder = remainder % (n_bom * n_temp * n_seed)

    bom_idx = remainder // (n_temp * n_seed)
    remainder = remainder % (n_temp * n_seed)

    proposal_temp_idx = remainder // n_seed
    seed_idx = remainder % n_seed

    return {
        "proposal_idx": proposal_idx,
        "reviewer_idx": reviewer_idx,
        "bom_idx": bom_idx,
        "proposal_temp_idx": proposal_temp_idx,
        "seed_idx": seed_idx,
        "proposal_prompt": proposal_prompts[proposal_idx],
        "reviewer_prompt": reviewer_prompts[reviewer_idx],
        "bom_variant": bom_variants[bom_idx],
        "proposal_temperature": proposal_temperatures[proposal_temp_idx],
        "seed": seed_values[seed_idx],
        "total": total,
    }


def main():
    args = parse_args()

    if args.model_name:
        os.environ["LOCAL_LLM_MODEL"] = args.model_name
    if args.embed_model:
        os.environ["LOCAL_EMBED_MODEL"] = args.embed_model

    from proposal_generator import run_rag1
    from scientific_advisor import run_rag2

    proposal_prompts = load_prompt_variants(args.proposal_prompts_file, "proposal")
    reviewer_prompts = load_prompt_variants(args.reviewer_prompts_file, "reviewer")
    bom_variants = load_bom_variants(args.bom_file)
    proposal_temperatures = parse_float_list(args.proposal_temperatures)
    seed_values = parse_int_list(args.seed_values)

    task_id = resolve_task_id(args.task_id)
    combo = select_combo(
        task_id=task_id,
        proposal_prompts=proposal_prompts,
        reviewer_prompts=reviewer_prompts,
        bom_variants=bom_variants,
        proposal_temperatures=proposal_temperatures,
        seed_values=seed_values,
    )

    proposal_idx = combo["proposal_idx"]
    reviewer_idx = combo["reviewer_idx"]
    bom_idx = combo["bom_idx"]
    proposal_temp_idx = combo["proposal_temp_idx"]
    seed_idx = combo["seed_idx"]
    proposal_variant = combo["proposal_prompt"]
    reviewer_variant = combo["reviewer_prompt"]
    bom_variant = combo["bom_variant"]
    proposal_temperature = combo["proposal_temperature"]
    seed = combo["seed"]

    run_name = (
        f"task_{task_id:03d}"
        f"_proposal_{proposal_idx:02d}"
        f"_reviewer_{reviewer_idx:02d}"
        f"_bom_{bom_idx:02d}"
        f"_ptemp_{proposal_temp_idx:02d}"
        f"_seed_{seed_idx:02d}"
    )

    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "task_id": task_id,
        "proposal_prompt_idx": proposal_idx,
        "proposal_prompt_name": proposal_variant["name"],
        "proposal_prompt_text": proposal_variant["text"],
        "reviewer_prompt_idx": reviewer_idx,
        "reviewer_prompt_name": reviewer_variant["name"],
        "reviewer_prompt_text": reviewer_variant["text"],
        "bom_variant_idx": bom_idx,
        "bom_variant_name": bom_variant["name"],
        "bom_variant": bom_variant["bom"],
        "proposal_temperature_idx": proposal_temp_idx,
        "proposal_temperature": proposal_temperature,
        "reviewer_temperature": args.reviewer_temperature,
        "seed_idx": seed_idx,
        "seed": seed,
        "top_p": args.top_p,
        "experiment_index_dir": str(args.experiment_index_dir.resolve()),
        "science_index_dir": str(args.science_index_dir.resolve()),
        "chunks_labeled_path": str(args.chunks_labeled_path.resolve()),
        "max_rounds": args.max_rounds,
        "model_name": os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen3-8B"),
        "embed_model": os.environ.get("LOCAL_EMBED_MODEL", ""),
        "n_proposal_prompts": len(proposal_prompts),
        "n_reviewer_prompts": len(reviewer_prompts),
        "n_bom_variants": len(bom_variants),
        "n_proposal_temperatures": len(proposal_temperatures),
        "n_seed_values": len(seed_values),
        "total_combinations": combo["total"],
    }

    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    (run_dir / "selected_proposal_prompt.txt").write_text(
        proposal_variant["text"], encoding="utf-8"
    )
    (run_dir / "selected_reviewer_prompt.txt").write_text(
        reviewer_variant["text"], encoding="utf-8"
    )
    (run_dir / "selected_bom_variant.txt").write_text(
        json.dumps(bom_variant["bom"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rag2_feedback = None
    rag1_output = None
    rag2_advice = None
    cached_exp_evidence = None
    cached_sci_evidence = None

    for round_idx in range(args.max_rounds):
        print("\n" + "=" * 100)
        print(f"MULTI-AGENT ROUND {round_idx + 1}")
        print("=" * 100)

        rag1_output = run_rag1(
            rag2_feedback=rag2_feedback,
            cached_exp_evidence=cached_exp_evidence,
            save_output=True,
            available_bom=bom_variant["bom"],
            prompt_variant_text=proposal_variant["text"],
            experiment_index_dir=args.experiment_index_dir,
            chunks_labeled_path=args.chunks_labeled_path,
            output_path=run_dir / f"rag1_round_{round_idx + 1}.json",
            temperature=proposal_temperature,
            top_p=args.top_p,
            seed=seed,
            model_name=args.model_name,
        )

        if cached_exp_evidence is None:
            cached_exp_evidence = rag1_output.get("cached_exp_evidence")

        rag2_result = run_rag2(
            rag1_output=rag1_output,
            cached_sci_evidence=cached_sci_evidence,
            save_output=True,
            prompt_variant_text=reviewer_variant["text"],
            science_index_dir=args.science_index_dir,
            chunks_labeled_path=args.chunks_labeled_path,
            output_path=run_dir / "rag2_advice.json",
            temperature=args.reviewer_temperature,
            top_p=args.top_p,
            seed=seed,
            model_name=args.model_name,
        )

        rag2_advice = rag2_result["rag2_advice"]

        if cached_sci_evidence is None:
            cached_sci_evidence = rag2_result["cached_sci_evidence"]

        print_round_summary(round_idx, rag1_output, rag2_advice)

        if should_stop(rag2_advice, round_idx):
            print("\nStopping condition met.")
            break

        rag2_feedback = rag2_advice

    final_payload = {
        "run_metadata": metadata,
        "final_rag1_output": rag1_output,
        "final_rag2_advice": rag2_advice,
    }

    with (run_dir / "final_output.json").open("w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("MULTI-AGENT LOOP FINISHED")
    print("=" * 100)

    return final_payload


if __name__ == "__main__":
    main()
