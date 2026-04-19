from __future__ import annotations

import argparse
import json
import re
import signal
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings


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


# =========================================================
# Local Qwen LLM (no Ollama)
# =========================================================
class LocalQwenLLM:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        self.model_name = model_name
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def complete(self, prompt: str, max_new_tokens: int = 512) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True)


# =========================================================
# Keyword priors
# =========================================================
EXPERIMENT_KEYWORDS = [
    "experimental procedure",
    "experimental setup",
    "materials and methods",
    "welding procedure",
    "process parameters",
    "sample preparation",
    "specimen preparation",
    "heat input",
    "cooling rate",
    "current",
    "voltage",
    "travel speed",
    "shielding gas",
    "interpass temperature",
    "preheat",
    "post weld heat treatment",
    "filler metal",
    "base metal",
    "equipment",
    "welding machine",
    "wire feeder",
    "torch",
    "thermocouple",
    "optical microscope",
    "SEM",
    "EDS",
    "microhardness",
    "tensile test",
    "impact test",
    "charpy",
]

SCIENCE_KEYWORDS = [
    "mechanism",
    "microstructure evolution",
    "phase transformation",
    "transformation behavior",
    "grain boundary ferrite",
    "acicular ferrite",
    "bainite",
    "martensite",
    "M/A constituent",
    "segregation",
    "diffusion",
    "nucleation",
    "growth",
    "thermodynamics",
    "kinetics",
    "strengthening mechanism",
    "fracture mechanism",
    "property relationship",
    "microstructure-property relationship",
    "causal factor",
    "attributed to",
    "due to the",
    "because of",
    "resulted in",
    "led to",
    "promoted",
    "suppressed",
    "facilitated",
    "associated with",
    "correlated with",
    "can be explained by",
    "is explained by",
    "is related to",
    "contributed to",
    "formation of",
    "refinement of",
    "coarsening of",
    "microstructural change",
    "microstructural evolution",
    "toughness improvement",
    "hardness increase",
    "fracture surface",
]

SCIENCE_BONUS_TERMS = [
    "mechanism",
    "microstructure",
    "phase transformation",
    "fracture surface",
    "microstructure-property relationship",
    "strengthening mechanism",
    "fracture mechanism",
]

EXPERIMENT_CARD_TEMPLATE = {
    "process_families": [],
    "material_families": [],
    "material_systems": [],
    "equipment": [],
    "consumables": [],
    "controllable_parameters": [],
    "measurements_outputs": [],
    "joint_or_sample_types": [],
    "weldability_quality_factors": [],
    "common_constraints": [],
    "literature_experiment_summary": "",
}

SCIENCE_CARD_TEMPLATE = {
    "scientific_reasoning": "",
    "key_variables": [],
    "observed_trends": [],
    "scientific_hypotheses": [],
    "literature_science_summary": "",
}


# =========================================================
# Args
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-level experiment/science cards from existing chunks.jsonl"
    )
    parser.add_argument("--chunks-path", type=Path, default=Path("outputs/chunks.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-chunks-per-card", type=int, default=4)
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=None,
        help="Maximum seconds to allow the script to run before exiting cleanly with partial progress saved.",
    )
    return parser.parse_args()


# =========================================================
# IO utils
# =========================================================
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_key_from_metadata(metadata: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(metadata.get("source_path", "")),
        int(metadata.get("chunk_id", -1)),
        int(metadata.get("chunk_start", -1)),
        int(metadata.get("chunk_end", -1)),
    )


def paper_key_from_metadata(metadata: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(metadata.get("source_path", "")),
        str(metadata.get("title", "")),
        str(metadata.get("doi", "")),
    )


def chunk_row_from_document(d: Document) -> dict[str, Any]:
    return {
        "source_path": d.metadata.get("source_path", ""),
        "file_name": d.metadata.get("file_name", ""),
        "title": d.metadata.get("title", ""),
        "doi": d.metadata.get("doi", ""),
        "chunk_id": d.metadata.get("chunk_id", -1),
        "chunk_start": d.metadata.get("chunk_start", -1),
        "chunk_end": d.metadata.get("chunk_end", -1),
        "char_length": d.metadata.get("char_length", 0),
        "chunk_label": d.metadata.get("chunk_label", ""),
        "chunk_confidence": d.metadata.get("chunk_confidence", 0.0),
        "chunk_reason": d.metadata.get("chunk_reason", ""),
        "experiment_score": d.metadata.get("experiment_score", 0),
        "science_score": d.metadata.get("science_score", 0),
        "text": d.text,
    }


def load_processed_chunk_keys(path: Path) -> set[tuple[str, int, int, int]]:
    if not path.exists():
        return set()

    processed: set[tuple[str, int, int, int]] = set()
    for row in read_jsonl(path):
        processed.add(
            (
                str(row.get("source_path", "")),
                int(row.get("chunk_id", -1)),
                int(row.get("chunk_start", -1)),
                int(row.get("chunk_end", -1)),
            )
        )
    return processed


def load_processed_paper_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()

    processed: set[tuple[str, str, str]] = set()
    for row in read_jsonl(path):
        processed.add(
            (
                str(row.get("source_path", "")),
                str(row.get("source_title", "")),
                str(row.get("doi", "")),
            )
        )
    return processed


def write_progress_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def load_documents_from_chunks(chunks_path: Path) -> list[Document]:
    rows = read_jsonl(chunks_path)
    documents: list[Document] = []

    for row in rows:
        text = row.get("text", "")
        metadata = {
            "source_path": row.get("source_path", ""),
            "file_name": row.get("file_name", ""),
            "title": row.get("title", ""),
            "doi": row.get("doi", ""),
            "chunk_id": row.get("chunk_id", -1),
            "chunk_start": row.get("chunk_start", -1),
            "chunk_end": row.get("chunk_end", -1),
            "char_length": row.get("char_length", len(text)),
        }
        documents.append(Document(text=text, metadata=metadata))

    return documents


# =========================================================
# Prompt / JSON helpers
# =========================================================
def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    text = text.strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

    obj, _ = decoder.raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object.")
    return obj


def normalize_list_field(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v).strip()]


def retry_json_prompt(
    llm: LocalQwenLLM,
    prompt: str,
    max_attempts: int = 3,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    last_error = None
    last_raw_text = ""

    for attempt in range(max_attempts):
        final_prompt = prompt
        if attempt > 0:
            final_prompt += (
                "\n\nReturn ONLY one valid JSON object. "
                "No markdown. No explanation. No trailing text."
            )

        try:
            raw_text = llm.complete(final_prompt, max_new_tokens=max_new_tokens)
            last_raw_text = raw_text
            return extract_json_object(raw_text)
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Failed to parse JSON: {last_error}\nRAW:\n{last_raw_text[:1200]}")
    


# =========================================================
# Labeling
# =========================================================
def keyword_score(text: str, keywords: list[str]) -> int:
    text_l = (text or "").lower()
    score = 0

    for kw in keywords:
        kw_l = kw.lower()
        if kw_l in text_l:
            score += 2
        else:
            parts = [p for p in re.split(r"[^a-z0-9]+", kw_l) if len(p) > 2]
            hits = sum(1 for p in parts if p in text_l)
            if hits >= max(1, len(parts) // 2):
                score += 1
    return score


def science_bonus_score(text: str) -> int:
    text_l = (text or "").lower()
    bonus = 0
    for term in SCIENCE_BONUS_TERMS:
        if term in text_l:
            bonus += 1
    return bonus


def classify_chunk_type(llm: LocalQwenLLM, text: str, exp_score: int, sci_score: int) -> dict[str, Any]:
    prompt = f"""
You are classifying one chunk from welding literature.

Classify the chunk into exactly one of:
- experiment
- science
- both
- neither

Definitions:
- experiment = setup, procedure, materials, equipment, consumables, process parameters, testing protocol, measurement protocol
- science = mechanism, interpretation, theory, causal explanation, microstructure-property reasoning
- both = substantial content from both
- neither = references, acknowledgements, publisher text, generic intro, weakly relevant text

Weak prior:
experiment_score = {exp_score}
science_score = {sci_score}

Chunk:
\"\"\"
{text[:1800]}
\"\"\"

Return ONLY valid JSON:
{{
  "label": "experiment",
  "confidence": 0.0,
  "reason": ""
}}
"""
    data = retry_json_prompt(llm, prompt, max_attempts=2, max_new_tokens=160)

    label = str(data.get("label", "neither")).strip().lower()
    if label not in {"experiment", "science", "both", "neither"}:
        label = "neither"

    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    return {
        "label": label,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(data.get("reason", "")).strip(),
    }


def annotate_chunks(
    documents: list[Document],
    llm: LocalQwenLLM,
    *,
    labeled_path: Path | None = None,
    processed_keys: set[tuple[str, int, int, int]] | None = None,
    deadline_epoch: float | None = None,
) -> tuple[int, bool]:
    processed_count = 0

    for i, d in enumerate(documents, start=1):
        chunk_key = chunk_key_from_metadata(d.metadata)
        if processed_keys is not None and chunk_key in processed_keys:
            print(f"[Chunk {i}/{len(documents)}] skipping already labeled chunk")
            continue

        if should_stop(deadline_epoch):
            print("Stopping before next chunk due to timeout or termination request.")
            return processed_count, True

        text = d.text or ""
        exp_score = keyword_score(text, EXPERIMENT_KEYWORDS)
        sci_score = keyword_score(text, SCIENCE_KEYWORDS)
        sci_score += science_bonus_score(text)

        if exp_score >= 9 and exp_score - sci_score >= 4:
            cls = {"label": "experiment", "confidence": 0.92, "reason": "experiment dominant"}
        elif sci_score >= 7 and sci_score - exp_score >= 3:
            cls = {"label": "science", "confidence": 0.92, "reason": "science dominant"}
        elif exp_score >= 7 and sci_score >= 5 and abs(exp_score - sci_score) <= 2:
            cls = {"label": "both", "confidence": 0.88, "reason": "balanced evidence for both"}
        elif exp_score <= 1 and sci_score <= 1:
            cls = {"label": "neither", "confidence": 0.75, "reason": "no strong evidence"}
        else:
            try:
                cls = classify_chunk_type(llm, text, exp_score, sci_score)
            except Exception as e:
                if exp_score >= 4 and sci_score >= 4:
                    fallback_label = "both"
                elif exp_score > sci_score and exp_score >= 3:
                    fallback_label = "experiment"
                elif sci_score > exp_score and sci_score >= 3:
                    fallback_label = "science"
                else:
                    fallback_label = "neither"

                cls = {
                    "label": fallback_label,
                    "confidence": 0.35,
                    "reason": f"classification_failed_fallback: {str(e)}",
                }

        d.metadata["chunk_label"] = cls["label"]
        d.metadata["chunk_confidence"] = cls["confidence"]
        d.metadata["chunk_reason"] = cls["reason"]
        d.metadata["experiment_score"] = exp_score
        d.metadata["science_score"] = sci_score

        print(
            f"[Chunk {i}/{len(documents)}] "
            f"label={cls['label']} conf={cls['confidence']:.2f} "
            f"exp_score={exp_score} sci_score={sci_score}"
        )
        if labeled_path is not None:
            append_jsonl(labeled_path, chunk_row_from_document(d))
        if processed_keys is not None:
            processed_keys.add(chunk_key)
        processed_count += 1

    return processed_count, False


def retry_json_prompt_prefill(
    llm: LocalQwenLLM,
    prompt: str,
    max_attempts: int = 3,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    last_error = None
    last_raw_text = ""

    for attempt in range(max_attempts):
        final_prompt = prompt
        if attempt > 0:
            final_prompt += "\nReturn ONLY JSON."

        try:
            # 모델이 설명부터 시작하지 못하게 아예 { 로 시작시킴
            raw_text = "{" + llm.complete(final_prompt + "\n\nJSON:\n{", max_new_tokens=max_new_tokens)
            last_raw_text = raw_text
            return extract_json_object(raw_text)
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Failed to parse JSON: {last_error}\nRAW:\n{last_raw_text[:1200]}")

def _extract_partial_json_fields(raw_text: str, template: dict[str, Any]) -> dict[str, Any]:
    """
    Salvage whatever fields are already present in a truncated JSON-like string.
    Works best when the model produced valid keys but got cut off near the end.
    """
    out: dict[str, Any] = {}

    for key, default in template.items():
        key_pat = re.escape(f'"{key}"')

        if isinstance(default, list):
            # capture list content until the first closing ] if present
            m = re.search(key_pat + r'\s*:\s*\[(.*?)\]', raw_text, flags=re.DOTALL)
            if m:
                body = m.group(1)
                items = re.findall(r'"([^"\n]+)"', body)
                out[key] = [x.strip() for x in items if x.strip()]
            else:
                # salvage partially written list items even if ] is missing
                m2 = re.search(key_pat + r'\s*:\s*\[(.*?)(?:,\s*"[A-Za-z_][A-Za-z0-9_]*"\s*:|$)', raw_text, flags=re.DOTALL)
                if m2:
                    body = m2.group(1)
                    items = re.findall(r'"([^"\n]+)"', body)
                    out[key] = [x.strip() for x in items if x.strip()]

        else:
            # normal full string
            m = re.search(key_pat + r'\s*:\s*"([^"]*)"', raw_text, flags=re.DOTALL)
            if m:
                out[key] = m.group(1).strip()
            else:
                # salvage truncated string until next field or end
                m2 = re.search(key_pat + r'\s*:\s*"(.*?)(?:,\s*"[A-Za-z_][A-Za-z0-9_]*"\s*:|$)', raw_text, flags=re.DOTALL)
                if m2:
                    val = m2.group(1).strip()
                    val = val.rstrip(',').rstrip('}').rstrip(']').strip()
                    out[key] = val

    return out


def _generate_card_with_salvage(
    llm: LocalQwenLLM,
    prompt: str,
    template: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    """
    1) Try strict JSON parse first
    2) If parse fails, salvage already-generated fields from raw text
    """
    try:
        card = retry_json_prompt_prefill(
            llm,
            prompt,
            max_attempts=3,
            max_new_tokens=max_new_tokens,
        )
        return sanitize_card(card, template)

    except Exception as e:
        msg = str(e)
        raw_text = ""
        raw_pos = msg.find("RAW:\n")
        if raw_pos != -1:
            raw_text = msg[raw_pos + len("RAW:\n"):]

        salvaged = _extract_partial_json_fields(raw_text, template)
        salvaged = sanitize_card(salvaged, template)

        if card_has_real_content(salvaged):
            print("    salvage=PARTIAL_JSON_RECOVERED")
            return salvaged

        raise


# =========================================================
# Card generation
# =========================================================
def build_evidence_text(chunks: list[Document], max_chunks: int = 4, max_chars_per_chunk: int = 1200) -> str:
    ranked = sorted(
        chunks,
        key=lambda d: (
            d.metadata.get("chunk_confidence", 0),
            d.metadata.get("experiment_score", 0) + d.metadata.get("science_score", 0),
        ),
        reverse=True,
    )

    selected = ranked[:max_chunks]
    blocks = []

    for i, d in enumerate(selected, 1):
        text = (d.text or "").strip()
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + " ..."

        blocks.append(
            f"""[Evidence Chunk {i}]
Title: {d.metadata.get("title", "")}
DOI: {d.metadata.get("doi", "")}
Chunk Label: {d.metadata.get("chunk_label")}
Chunk Confidence: {d.metadata.get("chunk_confidence")}
Chunk Start: {d.metadata.get("chunk_start")}
Chunk ID: {d.metadata.get("chunk_id")}

Text:
{text}"""
        )

    return "\n\n".join(blocks)


def card_has_real_content(card: dict[str, Any]) -> bool:
    for v in card.values():
        if isinstance(v, list) and len(v) > 0:
            return True
        if isinstance(v, str) and v.strip():
            return True
    return False


def sanitize_card(card: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    out = template.copy()

    for k in out:
        if k in card:
            out[k] = card[k]

    for k in out:
        if isinstance(template[k], list):
            out[k] = normalize_list_field(out[k])
        else:
            out[k] = str(out[k]).strip()

    return out


def build_experiment_card(llm: LocalQwenLLM, chunks: list[Document], max_chunks: int = 4) -> dict[str, Any]:
    evidence_text = build_evidence_text(
        chunks,
        max_chunks=max_chunks,
        max_chars_per_chunk=800,
    )

    prompt = f"""
You are creating ONE experiment card for ONE welding paper.

Use ONLY the evidence below.
Do not invent facts.
If a field is not supported by the evidence, leave it empty.
Keep lists concise and short.

Evidence:
{evidence_text}

Return ONLY valid JSON:
{{
  "process_families": [],
  "material_families": [],
  "material_systems": [],
  "equipment": [],
  "consumables": [],
  "controllable_parameters": [],
  "measurements_outputs": [],
  "joint_or_sample_types": [],
  "weldability_quality_factors": [],
  "common_constraints": [],
  "literature_experiment_summary": ""
}}
"""
    return _generate_card_with_salvage(
        llm=llm,
        prompt=prompt,
        template=EXPERIMENT_CARD_TEMPLATE,
        max_new_tokens=384,
    )


def build_science_card(llm: LocalQwenLLM, chunks: list[Document], max_chunks: int = 4) -> dict[str, Any]:
    evidence_text = build_evidence_text(
        chunks,
        max_chunks=max_chunks,
        max_chars_per_chunk=600,
    )

    prompt = f"""
You are creating ONE science card for ONE welding paper.

Use ONLY the evidence below.
Do not invent facts.
If a field is not supported by the evidence, leave it empty.
Keep scientific_reasoning under 80 words.
Keep each list to at most 5 short items.
Use short phrases, not long sentences.

Evidence:
{evidence_text}

Return ONLY valid JSON:
{{
  "scientific_reasoning": "",
  "key_variables": [],
  "observed_trends": [],
  "scientific_hypotheses": [],
  "literature_science_summary": ""
}}
"""
    return _generate_card_with_salvage(
        llm=llm,
        prompt=prompt,
        template=SCIENCE_CARD_TEMPLATE,
        max_new_tokens=512,
    )


# =========================================================
# Retrieval docs
# =========================================================
def experiment_card_to_doc(title: str, doi: str, card: dict[str, Any]) -> Document:
    text = f"""
Source Title: {title}
DOI: {doi}
Card Type: experiment

Process Families: {", ".join(card["process_families"])}
Material Families: {", ".join(card["material_families"])}
Material Systems: {", ".join(card["material_systems"])}
Equipment: {", ".join(card["equipment"])}
Consumables: {", ".join(card["consumables"])}
Controllable Parameters: {", ".join(card["controllable_parameters"])}
Measurements / Outputs: {", ".join(card["measurements_outputs"])}
Joint / Sample Types: {", ".join(card["joint_or_sample_types"])}
Weldability / Quality Factors: {", ".join(card["weldability_quality_factors"])}
Common Constraints: {", ".join(card["common_constraints"])}

Literature Experiment Summary:
{card["literature_experiment_summary"]}
""".strip()

    return Document(
        text=text,
        metadata={
            "card_type": "experiment",
            "source_title": title,
            "doi": doi,
        },
    )


def science_card_to_doc(title: str, doi: str, card: dict[str, Any]) -> Document:
    text = f"""
Source Title: {title}
DOI: {doi}
Card Type: science

Scientific Reasoning:
{card["scientific_reasoning"]}

Key Variables: {", ".join(card["key_variables"])}
Observed Trends: {", ".join(card["observed_trends"])}
Scientific Hypotheses: {", ".join(card["scientific_hypotheses"])}

Literature Science Summary:
{card["literature_science_summary"]}
""".strip()

    return Document(
        text=text,
        metadata={
            "card_type": "science",
            "source_title": title,
            "doi": doi,
        },
    )


def build_paper_memory_cards(
    documents: list[Document],
    llm: LocalQwenLLM,
    out_dir: Path,
    max_chunks: int = 4,
    build_index: bool = True,
    cards_path: Path | None = None,
    processed_paper_keys: set[tuple[str, str, str]] | None = None,
    deadline_epoch: float | None = None,
) -> list[dict[str, Any]]:
    paper_groups: dict[tuple[str, str, str], list[Document]] = defaultdict(list)

    for d in documents:
        key = (
            d.metadata.get("source_path", ""),
            d.metadata.get("title", ""),
            d.metadata.get("doi", ""),
        )
        paper_groups[key].append(d)

    outputs = []
    card_docs = [] if build_index else None

    cards_path = cards_path or (out_dir / "paper_memory_cards.jsonl")

    for idx, ((source_path, title, doi), chunk_docs) in enumerate(paper_groups.items(), start=1):
        paper_key = (source_path, title, doi)
        if processed_paper_keys is not None and paper_key in processed_paper_keys:
            print(f"\n[Paper {idx}/{len(paper_groups)}] skipping already saved {title}")
            continue

        if should_stop(deadline_epoch):
            print("Stopping before next paper due to timeout or termination request.")
            break

        print(f"\n[Paper {idx}/{len(paper_groups)}] {title}")

        exp_chunks = []
        sci_chunks = []

        for d in chunk_docs:
            label = d.metadata.get("chunk_label", "neither")
            if label in {"experiment", "both"}:
                exp_chunks.append(d)
            if label in {"science", "both"}:
                sci_chunks.append(d)

        print(f"  total_chunks={len(chunk_docs)}")
        print(f"  exp_candidate_chunks={len(exp_chunks)}")
        print(f"  sci_candidate_chunks={len(sci_chunks)}")

        row = {
            "source_path": source_path,
            "source_title": title,
            "doi": doi,
        }

        if exp_chunks:
            try:
                exp_card = build_experiment_card(llm, exp_chunks, max_chunks=max_chunks)
                if card_has_real_content(exp_card):
                    row["experiment_card"] = exp_card
                    if build_index:
                        card_docs.append(experiment_card_to_doc(title, doi, exp_card))
                    print("  experiment_card=CREATED")

                    preview = (
                        exp_card.get("literature_experiment_summary", "")[:200]
                        .replace("\n", " ")
                        .strip()
                    )
                    print(f"    preview={preview}")
                else:
                    print("  experiment_card=EMPTY")
            except Exception as e:
                row["experiment_card_error"] = str(e)
                print(f"  experiment_card=ERROR: {e}")
        else:
            print("  experiment_card=SKIPPED (no experiment/both chunks)")

        if sci_chunks:
            try:
                sci_card = build_science_card(llm, sci_chunks, max_chunks=max_chunks)
                if card_has_real_content(sci_card):
                    row["science_card"] = sci_card
                    if build_index:
                        card_docs.append(science_card_to_doc(title, doi, sci_card))
                    print("  science_card=CREATED")

                    preview = (
                        sci_card.get("literature_science_summary", "")[:200]
                        .replace("\n", " ")
                        .strip()
                    )
                    print(f"    preview={preview}")
                else:
                    print("  science_card=EMPTY")
            except Exception as e:
                row["science_card_error"] = str(e)
                print(f"  science_card=ERROR: {e}")
        else:
            print("  science_card=SKIPPED (no science/both chunks)")

        if (
            "experiment_card" in row
            or "science_card" in row
            or "experiment_card_error" in row
            or "science_card_error" in row
        ):
            outputs.append(row)

            # paper 하나 끝날 때마다 바로 append 저장
            append_jsonl(cards_path, row)
            if processed_paper_keys is not None:
                processed_paper_keys.add(paper_key)

            print("  row=SAVED")
        else:
            print("  row=NOT_SAVED")

    print(f"\nSaved {len(outputs)} paper memory cards to {cards_path}")

    if build_index and card_docs:
        index = VectorStoreIndex.from_documents(card_docs, show_progress=True)
        index.storage_context.persist(persist_dir=str(out_dir / "paper_memory_storage"))
        print(f"Persisted vector store to {out_dir / 'paper_memory_storage'}")

    return outputs


# =========================================================
# Main
# =========================================================
def main() -> None:
    args = parse_args()
    chunks_path = args.chunks_path.resolve()
    out_dir = args.out_dir.resolve()
    signal.signal(signal.SIGTERM, _handle_term)
    deadline_epoch = time.time() + args.max_seconds if args.max_seconds is not None else None
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = out_dir / "chunks_labeled.jsonl"
    summary_path = out_dir / "chunks_labeled_summary.json"

    if not chunks_path.exists():
        raise SystemExit(f"chunks.jsonl does not exist: {chunks_path}")

    documents = load_documents_from_chunks(chunks_path)
    if not documents:
        raise SystemExit("No documents loaded from chunks.jsonl")

    llm = LocalQwenLLM(model_name=args.model_name)

    Settings.embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    processed_keys = load_processed_chunk_keys(labeled_path)
    newly_processed, stopped_early = annotate_chunks(
        documents,
        llm,
        labeled_path=labeled_path,
        processed_keys=processed_keys,
        deadline_epoch=deadline_epoch,
    )

    summary = {
        "chunks_path": str(chunks_path),
        "chunks_labeled_path": str(labeled_path),
        "total_input_chunks": len(documents),
        "labeled_chunk_count": len(processed_keys),
        "newly_labeled_chunk_count": newly_processed,
        "completed_all_chunks": len(processed_keys) >= len(documents),
        "timed_out": stopped_early or should_stop(deadline_epoch),
        "stop_requested": STOP_REQUESTED,
    }
    write_progress_summary(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # build_paper_memory_cards(documents, llm, out_dir, max_chunks=args.max_chunks_per_card)


if __name__ == "__main__":
    main()
