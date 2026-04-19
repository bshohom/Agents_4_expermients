"""Build a persistent dense index from normalized paper-card JSONL.

Input format (one JSON object per line) should be roughly like:
{
  "source_title": "...",
  "doi": "...",
  "process_family": "...",
  "material_family": "...",
  "equipment": ["..."],
  "controllable_parameters": ["..."],
  "measurements_outputs": ["..."],
  "bom_keywords": ["..."],
  "experiment_summary": "...",
  "feasibility_notes": "..."
}

This builder converts each card into a retrieval text blob, embeds it with
Sentence Transformers, and writes:
- embeddings.npy
- records.jsonl
- manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    return p.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_join(x) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return ", ".join(str(v) for v in x)
    return str(x)


def card_to_text(card: Dict) -> str:
    """Turn a structured card into a retrieval-friendly text string.

    TODO for future coding agent:
    - optionally weight fields differently
    - optionally add repeated keywords for retrieval emphasis
    - optionally include section-level provenance
    """
    return (
        f"Title: {card.get('source_title', '')}\n"
        f"DOI: {card.get('doi', '')}\n"
        f"Process Family: {card.get('process_family', 'unknown')}\n"
        f"Material Family: {card.get('material_family', 'unknown')}\n"
        f"Material System: {safe_join(card.get('material_system', []))}\n"
        f"Equipment: {safe_join(card.get('equipment', []))}\n"
        f"Consumables: {safe_join(card.get('consumables', []))}\n"
        f"Controllable Parameters: {safe_join(card.get('controllable_parameters', []))}\n"
        f"Measurements / Outputs: {safe_join(card.get('measurements_outputs', []))}\n"
        f"BOM Keywords: {safe_join(card.get('bom_keywords', []))}\n"
        f"Experiment Summary: {card.get('experiment_summary', '')}\n"
        f"Feasibility Notes: {card.get('feasibility_notes', '')}\n"
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cards = load_jsonl(input_path)
    if not cards:
        raise ValueError(f"No cards found in {input_path}")

    texts = [card_to_text(card) for card in cards]
    model = SentenceTransformer(args.embedding_model)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    np.save(output_dir / "embeddings.npy", embeddings)

    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for card, text in zip(cards, texts):
            row = dict(card)
            row["_retrieval_text"] = text
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "embedding_model_name": args.embedding_model,
        "num_records": len(cards),
        "embedding_dim": int(embeddings.shape[1]),
        "source_jsonl": str(input_path),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote index to: {output_dir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
