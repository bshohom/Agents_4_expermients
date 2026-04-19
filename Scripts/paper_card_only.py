from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import torch
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings

from cards_divider import (
    LocalQwenLLM,
    build_paper_memory_cards,
    load_processed_paper_keys,
)


STOP_REQUESTED = False


def _handle_term(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def should_stop(deadline_epoch: float | None) -> bool:
    if STOP_REQUESTED:
        return True
    if deadline_epoch is None:
        return False
    return time.time() >= deadline_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild paper memory cards from existing chunks_labeled.jsonl"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory containing chunks_labeled.jsonl and where new card/index outputs will be saved.",
    )
    parser.add_argument(
        "--chunks-labeled-path",
        type=Path,
        default=None,
        help="Optional explicit path to chunks_labeled.jsonl. If omitted, uses <out-dir>/chunks_labeled.jsonl",
    )
    parser.add_argument(
        "--cards-path",
        type=Path,
        default=None,
        help="Optional explicit path to paper_memory_cards.jsonl. If omitted, uses <out-dir>/paper_memory_cards.jsonl",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Local generation model name.",
    )
    parser.add_argument(
        "--max-chunks-per-card",
        type=int,
        default=5,
        help="Maximum number of labeled chunks to send into one card-building prompt.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of labeled chunks to load for debugging.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Only rebuild paper_memory_cards.jsonl and skip rebuilding vector indexes.",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=None,
        help="Maximum seconds to allow the script to run before exiting cleanly with partial progress saved.",
    )
    return parser.parse_args()


def load_labeled_documents(path: Path, limit: int | None = None) -> list[Document]:
    docs: list[Document] = []

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break

            row = json.loads(line)
            d = Document(
                text=row.get("text", ""),
                metadata={
                    "source_path": row.get("source_path", ""),
                    "file_name": row.get("file_name", ""),
                    "title": row.get("title", ""),
                    "doi": row.get("doi", ""),
                    "chunk_id": row.get("chunk_id", -1),
                    "chunk_start": row.get("chunk_start", -1),
                    "chunk_end": row.get("chunk_end", -1),
                    "char_length": row.get("char_length", len(row.get("text", ""))),
                    "chunk_label": row.get("chunk_label", "neither"),
                    "chunk_confidence": row.get("chunk_confidence", 0.0),
                    "chunk_reason": row.get("chunk_reason", ""),
                    "experiment_score": row.get("experiment_score", 0),
                    "science_score": row.get("science_score", 0),
                },
            )
            docs.append(d)

    return docs


def normalize_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v).strip()]


def card_has_real_content(card):
    if not card:
        return False

    for v in card.values():
        if isinstance(v, list) and len(v) > 0:
            return True
        if isinstance(v, str) and v.strip():
            return True

    return False


def first_or_unknown(values):
    return values[0] if values else "unknown"


def load_experiment_card_docs(cards_path: Path) -> list[Document]:
    docs: list[Document] = []

    with cards_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            exp_card = row.get("experiment_card")

            if not card_has_real_content(exp_card):
                continue

            process_families = normalize_list(exp_card.get("process_families"))
            material_families = normalize_list(exp_card.get("material_families"))
            material_systems = normalize_list(exp_card.get("material_systems"))

            text = f"""
Source Title: {row.get('source_title', '')}
DOI: {row.get('doi', '')}
Card Type: experiment

Process Families: {", ".join(process_families)}
Material Families: {", ".join(material_families)}
Material Systems: {", ".join(material_systems)}
Equipment: {", ".join(normalize_list(exp_card.get("equipment")))}
Consumables: {", ".join(normalize_list(exp_card.get("consumables")))}
Controllable Parameters: {", ".join(normalize_list(exp_card.get("controllable_parameters")))}
Measurements / Outputs: {", ".join(normalize_list(exp_card.get("measurements_outputs")))}
Joint / Sample Types: {", ".join(normalize_list(exp_card.get("joint_or_sample_types")))}
Weldability / Quality Factors: {", ".join(normalize_list(exp_card.get("weldability_quality_factors")))}
Common Constraints: {", ".join(normalize_list(exp_card.get("common_constraints")))}

Literature Experiment Summary:
{(exp_card.get("literature_experiment_summary") or "").strip()}
""".strip()

            docs.append(
                Document(
                    text=text,
                    metadata={
                        "card_type": "experiment",
                        "source_title": row.get("source_title", ""),
                        "doi": row.get("doi", ""),
                        "source_path": row.get("source_path", ""),
                        "process_family": first_or_unknown(process_families),
                        "material_family": first_or_unknown(material_families),
                        "process_families": process_families,
                        "material_families": material_families,
                        "material_systems": material_systems,
                    },
                )
            )

    return docs


def load_science_card_docs(cards_path: Path) -> list[Document]:
    docs: list[Document] = []

    with cards_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sci_card = row.get("science_card")

            if not card_has_real_content(sci_card):
                continue

            text = f"""
Source Title: {row.get('source_title', '')}
DOI: {row.get('doi', '')}
Card Type: science

Scientific Reasoning:
{(sci_card.get("scientific_reasoning") or "").strip()}

Key Variables: {", ".join(normalize_list(sci_card.get("key_variables")))}
Observed Trends: {", ".join(normalize_list(sci_card.get("observed_trends")))}
Scientific Hypotheses: {", ".join(normalize_list(sci_card.get("scientific_hypotheses")))}

Literature Science Summary:
{(sci_card.get("literature_science_summary") or "").strip()}
""".strip()

            docs.append(
                Document(
                    text=text,
                    metadata={
                        "card_type": "science",
                        "source_title": row.get("source_title", ""),
                        "doi": row.get("doi", ""),
                        "source_path": row.get("source_path", ""),
                    },
                )
            )

    return docs


def build_indexes(out_dir: Path, cards_path: Path) -> None:
    print("Loading experiment cards...", flush=True)
    exp_docs = load_experiment_card_docs(cards_path)
    print(f"Loaded {len(exp_docs)} experiment docs", flush=True)

    print("Loading science cards...", flush=True)
    sci_docs = load_science_card_docs(cards_path)
    print(f"Loaded {len(sci_docs)} science docs", flush=True)

    if exp_docs:
        exp_index = VectorStoreIndex.from_documents(exp_docs, show_progress=True)
        exp_index.storage_context.persist(
            persist_dir=str(out_dir / "paper_memory_storage_experiment")
        )
        print("Saved experiment index", flush=True)
    else:
        print("No usable experiment docs found; skipped experiment index.", flush=True)

    if sci_docs:
        sci_index = VectorStoreIndex.from_documents(sci_docs, show_progress=True)
        sci_index.storage_context.persist(
            persist_dir=str(out_dir / "paper_memory_storage_science")
        )
        print("Saved science index", flush=True)
    else:
        print("No usable science docs found; skipped science index.", flush=True)


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, _handle_term)
    deadline_epoch = time.time() + args.max_seconds if args.max_seconds is not None else None

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "paper_memory_cards_summary.json"

    chunks_labeled_path = (
        args.chunks_labeled_path.resolve()
        if args.chunks_labeled_path is not None
        else (out_dir / "chunks_labeled.jsonl").resolve()
    )

    cards_path = (
        args.cards_path.resolve()
        if args.cards_path is not None
        else (out_dir / "paper_memory_cards.jsonl").resolve()
    )

    if not chunks_labeled_path.exists():
        raise FileNotFoundError(f"Missing chunks_labeled file: {chunks_labeled_path}")

    Settings.embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    print("Loading labeled chunks...", flush=True)
    docs = load_labeled_documents(chunks_labeled_path, limit=args.limit)
    print(f"Loaded {len(docs)} labeled chunks", flush=True)

    processed_paper_keys = load_processed_paper_keys(cards_path)
    total_input_papers = len(
        {
            (
                d.metadata.get("source_path", ""),
                d.metadata.get("title", ""),
                d.metadata.get("doi", ""),
            )
            for d in docs
        }
    )

    print("Building paper memory cards...", flush=True)
    llm = LocalQwenLLM(args.model_name)

    outputs = build_paper_memory_cards(
        docs,
        llm,
        out_dir,
        max_chunks=args.max_chunks_per_card,
        build_index=False,
        cards_path=cards_path,
        processed_paper_keys=processed_paper_keys,
        deadline_epoch=deadline_epoch,
    )
    print(f"Rebuilt {len(outputs)} paper memory card rows", flush=True)

    if not cards_path.exists():
        raise FileNotFoundError(f"paper_memory_cards.jsonl was not created: {cards_path}")

    timed_out = should_stop(deadline_epoch) or STOP_REQUESTED

    summary = {
        "chunks_labeled_path": str(chunks_labeled_path),
        "cards_path": str(cards_path),
        "total_input_papers": total_input_papers,
        "saved_paper_count": len(load_processed_paper_keys(cards_path)),
        "newly_saved_paper_count": len(outputs),
        "completed_all_papers": len(load_processed_paper_keys(cards_path)) >= total_input_papers,
        "timed_out": timed_out,
        "stop_requested": STOP_REQUESTED,
        "skip_index": args.skip_index,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.skip_index:
        print("Skipped vector index rebuild because --skip-index was set.", flush=True)
        print("Done.", flush=True)
        return

    if timed_out:
        print("Skipping vector index rebuild because the run stopped early.", flush=True)
        print("Done.", flush=True)
        return

    build_indexes(out_dir, cards_path)
    summary["experiment_index_dir"] = str(out_dir / "paper_memory_storage_experiment")
    summary["science_index_dir"] = str(out_dir / "paper_memory_storage_science")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
