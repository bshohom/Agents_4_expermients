from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build experiment/science indexes from paper_memory_cards.jsonl")
    parser.add_argument("--cards-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--embed-model", type=str, default=None)
    return parser.parse_args()


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
    docs = []
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
    docs = []
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


def main():
    args = parse_args()

    embed_model_name = args.embed_model or os.environ.get("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    Settings.embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name=embed_model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    exp_docs = load_experiment_card_docs(args.cards_path)
    sci_docs = load_science_card_docs(args.cards_path)

    print(f"Loaded {len(exp_docs)} experiment docs")
    print(f"Loaded {len(sci_docs)} science docs")

    if exp_docs:
        exp_index = VectorStoreIndex.from_documents(exp_docs, show_progress=True)
        exp_index.storage_context.persist(
            persist_dir=str(args.out_dir / "paper_memory_storage_experiment")
        )
        print("Saved experiment index")

    if sci_docs:
        sci_index = VectorStoreIndex.from_documents(sci_docs, show_progress=True)
        sci_index.storage_context.persist(
            persist_dir=str(args.out_dir / "paper_memory_storage_science")
        )
        print("Saved science index")


if __name__ == "__main__":
    main()