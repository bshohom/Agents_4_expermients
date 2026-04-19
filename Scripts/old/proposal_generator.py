import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.core import Settings, load_index_from_storage, StorageContext
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# Simple local LLM wrapper
# =========================================================
class LocalQwenLLM:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def complete(self, prompt: str, max_new_tokens: int = 550) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        if "<think>" in text.lower():
            idx = text.lower().find("candidate experiment:")
            if idx != -1:
                text = text[idx:].strip()

        return text


# =========================================================
# Embedding model
# =========================================================
LOCAL_EMBED_MODEL = ".../huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

Settings.embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(
        model_name=LOCAL_EMBED_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
)


# =========================================================
# Shared objects / constants
# =========================================================
llm = LocalQwenLLM("Qwen/Qwen3-8B")

AVAILABLE_BOM = {
    "material_family": "metals",
    "process_family": "joining",
    "materials": [
        "steel",
        "stainless steel",
        "aluminum alloy",
    ],
    "equipment": [
        "welding machine",
        "thermocouple",
        "microhardness tester",
        "tensile testing machine",
        "optical microscope",
        "sample cutting tool",
        "metallographic polishing kit",
        "pressure control module",
    ],
    "forbidden_items": [
        "EBSD",
        "synchrotron",
        "CFD simulation",
    ],
    "goal": "Propose one feasible first-pass joining experiment on metallic materials."
}

EXPERIMENT_STORAGE_DIR = "outputs/paper_memory_storage_experiment"
CHUNKS_LABELED_PATH = Path("outputs/chunks_labeled.jsonl")
RAG1_OUTPUT_PATH = Path("outputs/rag1_latest_output.json")


# =========================================================
# Helpers
# =========================================================
def overlap_score_text(text: str, bom: dict) -> int:
    text = (text or "").lower()
    score = 0

    for item in bom["materials"]:
        if item.lower() in text:
            score += 2

    for item in bom["equipment"]:
        if item.lower() in text:
            score += 2

    for item in bom["forbidden_items"]:
        if item.lower() in text:
            score -= 4

    return score


def overlap_score_node(node, bom: dict) -> int:
    return overlap_score_text(node.text or "", bom)


def load_labeled_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def recover_supporting_chunks(shortlist, labeled_rows: list[dict], bom: dict, top_chunks_per_paper: int = 2) -> dict:
    selected_source_paths = {
        node.metadata.get("source_path", "") for node in shortlist
    }

    paper_to_chunks = {}

    for source_path in selected_source_paths:
        candidate_rows = [
            row for row in labeled_rows
            if row.get("source_path", "") == source_path
            and row.get("chunk_label") in {"experiment", "both"}
        ]

        ranked_rows = sorted(
            candidate_rows,
            key=lambda row: (
                overlap_score_text(row.get("text", ""), bom),
                row.get("chunk_confidence", 0.0),
                row.get("experiment_score", 0) + row.get("science_score", 0),
            ),
            reverse=True,
        )

        paper_to_chunks[source_path] = ranked_rows[:top_chunks_per_paper]

    return paper_to_chunks


def trim_repeated_sections(text: str) -> str:
    marker = "Candidate Experiment:"
    first = text.find(marker)
    if first == -1:
        return text.strip()

    second = text.find(marker, first + len(marker))
    if second != -1:
        return text[:second].strip()

    return text.strip()


def build_rag1_prompt(
    available_bom: dict,
    card_context: str,
    chunk_context: str,
    rag2_feedback: dict | None = None,
) -> str:
    feedback_block = ""

    if rag2_feedback is not None:
        feedback_block = f"""

Revision Feedback From RAG2:
{json.dumps(rag2_feedback, indent=2)}

IMPORTANT:
- Apply only practical feasibility feedback.
- Keep the core experiment idea unless narrowing is necessary.
- Prefer reducing variables over expanding scope.
- Prefer a simpler first-step experiment that is easier to execute and interpret.
- Rewrite the full proposal so every section consistently reflects the revised experiment design after applying the feedback.
- Under "Changes From RAG2 Feedback", briefly state 2-4 concrete changes you made from the feedback.
- Ignore any RAG2 feedback or self-generated revision that introduces or assumes anything not explicitly listed in the BOM.
"""

    return f"""
You are proposing ONE candidate new experiment.

Use the literature cards as precedent summaries.
Use the supporting chunks as grounded evidence.
Do NOT pretend the experiment is already published.
You may synthesize across multiple papers.
You must respect the constrained BOM.
Answer in English only.
Do not output reasoning.
Do not output <think>.
Do not assume a specialized process, material subtype unless it is explicitly supported by the BOM.


Available BOM / Lab Capability:
{json.dumps(available_bom, indent=2)}

Literature Cards:
{card_context}

Supporting Evidence Chunks:
{chunk_context}{feedback_block}

Task:
Propose ONE plausible first-pass experiment that is inspired by the retrieved papers and compatible with the BOM.

Important:
- The experiment does not need to be fully optimized.
- Prefer a simple, executable first-step design.
- Do not use forbidden items.
- RAG2 will critique feasibility, missing items, and narrowing.


Return in this exact format:

Candidate Experiment:
Why This Is Feasible With Current BOM:
Borrowed Literature Precedents:
New Adaptation / Novel Twist:
Changes from RAG2 Feedback:
Needed Equipment:
Needed Materials / Consumables:
Key Process Parameters To Sweep:
Measurements / Outputs:
Main Risk / Failure Mode:
Missing Capability / Assumption:

Rules:
- Do not claim the experiment is novel with certainty.
- Distinguish clearly between literature-backed elements and your proposed adaptation.
- Do not include equipment or materials not supported by the BOM unless you put them under "Missing Capability / Assumption".
- If the BOM is insufficient, say so explicitly.
- Focus on proposing a clear first-pass experiment, not a fully optimized final protocol.
- Do not reintroduce any equipment, parameter, or geometry option that RAG2 explicitly asked you to remove or avoid.
- If RAG2 identifies an item as missing from the BOM, do not place it under Needed Equipment in the revision.
- Treat RAG2 narrowing advice as binding unless it directly conflicts with the BOM.


""".strip()


# =========================================================
# Main callable for orchestrator
# =========================================================
def run_rag1(
    rag2_feedback: dict | None = None,
    cached_exp_evidence: dict | None = None,
    save_output: bool = True,
) -> dict:
    available_bom = AVAILABLE_BOM

    # =====================================================
    # ROUND 1: retrieve and cache experiment evidence
    # =====================================================
    if cached_exp_evidence is None:
        storage_context = StorageContext.from_defaults(
            persist_dir=EXPERIMENT_STORAGE_DIR
        )
        paper_card_index = load_index_from_storage(storage_context)

        retriever = paper_card_index.as_retriever(similarity_top_k=8)
        retrieved_cards = retriever.retrieve(available_bom["goal"])

        filtered_cards = []

        target_process = available_bom["process_family"].lower()
        target_material = available_bom["material_family"].lower()

        for node in retrieved_cards:
            process_vals = " ".join(node.metadata.get("process_families", [])).lower()
            material_vals = " ".join(node.metadata.get("material_families", [])).lower()
            text = (node.text or "").lower()

            process_match = (
                target_process in process_vals
                or target_process in text
            )

            material_match = (
                target_material in material_vals
                or target_material in text
            )

            if process_match or material_match:
                filtered_cards.append(node)

        if len(filtered_cards) < 4:
            filtered_cards = list(retrieved_cards)

        ranked = sorted(
            filtered_cards,
            key=lambda n: overlap_score_node(n, available_bom),
            reverse=True,
        )

        shortlist = ranked[:7]

        print("RAG1 round-1 retrieval: caching experiment evidence pool.")
        print("Shortlisted paper cards:")
        for i, node in enumerate(shortlist, 1):
            print(f"\n[{i}] {node.metadata.get('source_title')}")
            print(f"process_families={node.metadata.get('process_families')}")
            print(f"material_families={node.metadata.get('material_families')}")

        labeled_rows = load_labeled_rows(CHUNKS_LABELED_PATH)
        paper_to_chunks = recover_supporting_chunks(
            shortlist=shortlist,
            labeled_rows=labeled_rows,
            bom=available_bom,
            top_chunks_per_paper=2,
        )

        print("Recovered supporting chunks:")
        for node in shortlist:
            source_path = node.metadata.get("source_path", "")
            title = node.metadata.get("source_title", "")
            n = len(paper_to_chunks.get(source_path, []))
            print(f"{title}: {n} chunks")

        cached_exp_evidence = {
            "cards": [
                {
                    "source_title": node.metadata.get("source_title", ""),
                    "doi": node.metadata.get("doi", ""),
                    "source_path": node.metadata.get("source_path", ""),
                    "text": node.text or "",
                    "process_families": node.metadata.get("process_families", []),
                    "material_families": node.metadata.get("material_families", []),
                }
                for node in shortlist
            ],
            "paper_to_chunks": paper_to_chunks,
        }

    # =====================================================
    # ROUND 2+: reuse cached experiment evidence only
    # =====================================================
    else:
        print("RAG1 is reusing cached experiment evidence pool.")
        shortlist = cached_exp_evidence["cards"]
        paper_to_chunks = cached_exp_evidence["paper_to_chunks"]

    # =====================================================
    # Build card context
    # =====================================================
    card_blocks = []
    for i, node in enumerate(shortlist, 1):
        if isinstance(node, dict):
            title = node.get("source_title", "")
            doi = node.get("doi", "")
            text = node.get("text", "")
        else:
            title = node.metadata.get("source_title", "")
            doi = node.metadata.get("doi", "")
            text = node.text or ""

        card_blocks.append(
            f"""[Paper Card {i}]
Title: {title}
DOI: {doi}

{text}
"""
        )

    card_context = "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(card_blocks)

    # =====================================================
    # Build chunk context
    # =====================================================
    chunk_blocks = []
    idx = 1
    source_titles = []

    for node in shortlist:
        if isinstance(node, dict):
            source_path = node.get("source_path", "")
            title = node.get("source_title", "")
        else:
            source_path = node.metadata.get("source_path", "")
            title = node.metadata.get("source_title", "")

        if title:
            source_titles.append(title)

        for row in paper_to_chunks.get(source_path, []):
            chunk_blocks.append(
                f"""[Supporting Chunk {idx}]
Title: {title}
Chunk ID: {row.get('chunk_id')}
Text:
{row.get('text', '')}
"""
            )
            idx += 1

    chunk_context = "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(chunk_blocks)

    # =====================================================
    # Build prompt
    # =====================================================
    if rag2_feedback is not None:
        print("RAG1 is revising with RAG2 feedback.")
    else:
        print("RAG1 is running first-round proposal generation.")

    proposal_prompt = build_rag1_prompt(
        available_bom=available_bom,
        card_context=card_context,
        chunk_context=chunk_context,
        rag2_feedback=rag2_feedback,
    )

    # =====================================================
    # Generate proposal
    # =====================================================
    proposal = llm.complete(proposal_prompt, max_new_tokens=550)

    if "Candidate Experiment:" in proposal:
        proposal = proposal[proposal.find("Candidate Experiment:"):].strip()

    proposal = trim_repeated_sections(proposal)

    spill_markers = ["\nOkay,", "\nLet me", "\nBased on the"]
    for marker in spill_markers:
        idx = proposal.find(marker)
        if idx != -1:
            proposal = proposal[:idx].rstrip()
            break

    proposal += "\n\nSource Papers Used:\n" + "\n".join(f"- {t}" for t in source_titles)

    print("\n" + "=" * 100)
    print("RAG1 PROPOSAL")
    print("=" * 100)
    print(proposal)

    rag1_output_payload = {
        "available_bom": available_bom,
        "rag1_proposal": proposal,
        "cached_exp_evidence": cached_exp_evidence,
    }

    if save_output:
        RAG1_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RAG1_OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(rag1_output_payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved RAG1 output to: {RAG1_OUTPUT_PATH}")

    return rag1_output_payload


if __name__ == "__main__":
    run_rag1(rag2_feedback=None, save_output=True)
