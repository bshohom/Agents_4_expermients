from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any
from zipfile import ZipFile, ZIP_DEFLATED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile prompt sweep outputs into blind judge packets and a manifest."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs_prompt_sweep"),
        help="Root directory containing task_* prompt sweep outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs_prompt_sweep_compiled"),
        help="Directory to write compiled judging artifacts.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Optional zip file path. Defaults to <output-root>.zip",
    )
    return parser.parse_args()


def safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def extract_task_id(run_dir: Path) -> int:
    m = re.match(r"task_(\d+)", run_dir.name)
    if not m:
        return -1
    return int(m.group(1))


def find_latest_rag1_round(run_dir: Path) -> Path | None:
    round_files = sorted(
        run_dir.glob("rag1_round_*.json"),
        key=lambda p: int(re.search(r"rag1_round_(\d+)\.json", p.name).group(1))
        if re.search(r"rag1_round_(\d+)\.json", p.name)
        else -1,
    )
    return round_files[-1] if round_files else None


def count_rounds(run_dir: Path) -> int:
    return len(list(run_dir.glob("rag1_round_*.json")))


def make_blind_plan_text(
    proposal_text: str,
    bom: dict[str, Any] | None,
    rag2_advice: dict[str, Any] | None,
) -> str:
    parts: list[str] = []

    if bom:
        parts.append("BOM SNAPSHOT")
        parts.append(json.dumps(bom, indent=2, ensure_ascii=False))

    parts.append("FINAL PROPOSAL")
    parts.append(proposal_text.strip() if proposal_text else "[MISSING PROPOSAL]")

    if rag2_advice:
        bom_check = rag2_advice.get("bom_check", {})
        narrowing = rag2_advice.get("narrowing_advice", [])

        parts.append("FINAL REVIEW SUMMARY")
        parts.append(
            json.dumps(
                {
                    "status": bom_check.get("status", ""),
                    "reason": bom_check.get("reason", ""),
                    "message_to_rag1": rag2_advice.get("message_to_rag1", ""),
                    "top_narrowing_advice": narrowing[:3],
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    return "\n\n" + ("\n\n" + "=" * 100 + "\n\n").join(parts) + "\n"


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "plan_id",
        "task_id",
        "run_dir",
        "proposal_prompt_idx",
        "proposal_prompt_name",
        "reviewer_prompt_idx",
        "reviewer_prompt_name",
        "bom_variant_idx",
        "bom_variant_name",
        "rounds_completed",
        "final_status",
        "final_reason",
        "parse_failed",
        "proposal_prompt_text",
        "reviewer_prompt_text",
        "bom_variant_text",
        "blind_txt_path",
        "blind_json_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def zip_directory(src_dir: Path, zip_path: Path) -> None:
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for file_path in src_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(src_dir.parent))


def main() -> None:
    args = parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    blind_root = output_root / "blind_packets"
    manifest_root = output_root / "manifests"

    blind_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(
        [p for p in input_root.iterdir() if p.is_dir() and p.name.startswith("task_")],
        key=lambda p: extract_task_id(p),
    )

    manifest_rows: list[dict[str, Any]] = []
    judge_jsonl_rows: list[dict[str, Any]] = []
    judge_csv_rows: list[dict[str, Any]] = []

    incomplete_runs: list[str] = []

    for blind_idx, run_dir in enumerate(run_dirs, start=1):
        plan_id = f"PLAN_{blind_idx:03d}"
        task_id = extract_task_id(run_dir)

        run_config = safe_read_json(run_dir / "run_config.json") or {}
        final_output = safe_read_json(run_dir / "final_output.json") or {}
        rag2_advice = (
            final_output.get("final_rag2_advice")
            or safe_read_json(run_dir / "rag2_advice.json")
            or {}
        )

        latest_rag1_path = find_latest_rag1_round(run_dir)
        latest_rag1 = safe_read_json(latest_rag1_path) if latest_rag1_path else {}
        latest_rag1 = latest_rag1 or final_output.get("final_rag1_output", {}) or {}

        proposal_text = latest_rag1.get("rag1_proposal", "")
        available_bom = latest_rag1.get("available_bom", None)

        proposal_prompt_text = safe_read_text(run_dir / "selected_proposal_prompt.txt")
        reviewer_prompt_text = safe_read_text(run_dir / "selected_reviewer_prompt.txt")
        bom_variant_text = safe_read_text(run_dir / "selected_bom_variant.txt")

        final_status = rag2_advice.get("bom_check", {}).get("status", "")
        final_reason = rag2_advice.get("bom_check", {}).get("reason", "")
        parse_failed = bool(rag2_advice.get("parse_failed", False))

        rounds_completed = count_rounds(run_dir)

        if not proposal_text:
            incomplete_runs.append(str(run_dir))

        blind_text = make_blind_plan_text(
            proposal_text=proposal_text,
            bom=available_bom,
            rag2_advice=rag2_advice,
        )

        blind_txt_path = blind_root / f"{plan_id}.txt"
        blind_json_path = blind_root / f"{plan_id}.json"

        blind_txt_path.write_text(blind_text, encoding="utf-8")
        with blind_json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "plan_id": plan_id,
                    "proposal_text": proposal_text,
                    "available_bom": available_bom,
                    "final_rag2_advice": rag2_advice,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        manifest_row = {
            "plan_id": plan_id,
            "task_id": task_id,
            "run_dir": str(run_dir),
            "proposal_prompt_idx": run_config.get("proposal_prompt_idx"),
            "proposal_prompt_name": run_config.get("proposal_prompt_name"),
            "reviewer_prompt_idx": run_config.get("reviewer_prompt_idx"),
            "reviewer_prompt_name": run_config.get("reviewer_prompt_name"),
            "bom_variant_idx": run_config.get("bom_variant_idx"),
            "bom_variant_name": run_config.get("bom_variant_name"),
            "rounds_completed": rounds_completed,
            "final_status": final_status,
            "final_reason": final_reason,
            "parse_failed": parse_failed,
            "proposal_prompt_text": proposal_prompt_text,
            "reviewer_prompt_text": reviewer_prompt_text,
            "bom_variant_text": bom_variant_text,
            "blind_txt_path": str(blind_txt_path),
            "blind_json_path": str(blind_json_path),
        }
        manifest_rows.append(manifest_row)

        judge_jsonl_rows.append(
            {
                "plan_id": plan_id,
                "proposal_text": proposal_text,
                "available_bom": available_bom,
                "final_status": final_status,
                "rounds_completed": rounds_completed,
            }
        )

        judge_csv_rows.append(
            {
                "plan_id": plan_id,
                "overall_score": "",
                "feasibility_score": "",
                "novelty_score": "",
                "clarity_score": "",
                "bom_alignment_score": "",
                "comments": "",
            }
        )

    with (manifest_root / "prompt_sweep_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, indent=2, ensure_ascii=False)

    with (manifest_root / "prompt_sweep_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_manifest_csv(manifest_root / "prompt_sweep_manifest.csv", manifest_rows)

    with (blind_root / "judge_input.jsonl").open("w", encoding="utf-8") as f:
        for row in judge_jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (blind_root / "judge_sheet.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "plan_id",
                "overall_score",
                "feasibility_score",
                "novelty_score",
                "clarity_score",
                "bom_alignment_score",
                "comments",
            ],
        )
        writer.writeheader()
        writer.writerows(judge_csv_rows)

    summary = {
        "n_runs_found": len(run_dirs),
        "n_incomplete_runs": len(incomplete_runs),
        "compiled_output_root": str(output_root),
        "blind_packets_dir": str(blind_root),
        "manifest_dir": str(manifest_root),
    }
    with (output_root / "compile_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    zip_path = args.zip_path.resolve() if args.zip_path else output_root.with_suffix(".zip")
    zip_directory(output_root, zip_path)

    print(json.dumps({**summary, "zip_path": str(zip_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()