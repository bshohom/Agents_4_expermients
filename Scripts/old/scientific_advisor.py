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

    def complete(self, prompt: str, max_new_tokens: int = 2500) -> str:
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
        return text


# =========================================================
# Embedding model
# =========================================================
LOCAL_EMBED_MODEL = "/trace/group/tmousavi/gyunghuy/cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

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

SCIENCE_STORAGE_DIR = "outputs/paper_memory_storage_science"
CHUNKS_LABELED_PATH = Path("outputs/chunks_labeled.jsonl")
RAG2_OUTPUT_PATH = Path("outputs/rag2_advice.json")


# =========================================================
# Helpers
# =========================================================
def overlap_score_text(text: str, bom: dict) -> int:
    text = (text or "").lower()
    score = 0

    for item in bom.get("materials", []):
        if item.lower() in text:
            score += 2

    for item in bom.get("equipment", []):
        if item.lower() in text:
            score += 2

    for item in bom.get("forbidden_items", []):
        if item.lower() in text:
            score -= 4

    process_family = bom.get("process_family", "")
    if process_family and process_family.lower() in text:
        score += 4

    material_family = bom.get("material_family", "")
    if material_family and material_family.lower() in text:
        score += 2

    return score


def load_labeled_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def recover_supporting_chunks(
    shortlist,
    labeled_rows: list[dict],
    bom: dict,
    top_chunks_per_paper: int = 3,
) -> dict:
    selected_source_paths = {
        node.metadata.get("source_path", "") for node in shortlist
    }

    paper_to_chunks = {}

    for source_path in selected_source_paths:
        candidate_rows = [
            row for row in labeled_rows
            if row.get("source_path", "") == source_path
            and row.get("chunk_label") in {"science", "both"}
        ]

        ranked_rows = sorted(
            candidate_rows,
            key=lambda row: (
                overlap_score_text(row.get("text", ""), bom),
                row.get("chunk_confidence", 0.0),
                row.get("science_score", 0) + row.get("experiment_score", 0),
            ),
            reverse=True,
        )

        paper_to_chunks[source_path] = ranked_rows[:top_chunks_per_paper]

    return paper_to_chunks


def build_rag2_query(bom: dict, proposal_text: str) -> str:
    return f"""
Process family: {bom.get("process_family", "")}
Material family: {bom.get("material_family", "")}
Goal: {bom.get("goal", "")}

RAG1 experiment proposal:
{proposal_text[:1000]}

Retrieve scientific literature that helps:
- verify whether this proposed experiment is feasible
- identify practical execution limits or capability constraints within the current BOM
- reduce the number of variables
- simplify the first experiment into a narrower, executable study
""".strip()


def extract_json_block(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found.")

    brace_count = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if not in_string:
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[start:i + 1]

    raise ValueError("No complete JSON object found.")


def build_rag2_prompt(
    available_bom: dict,
    rag1_proposal: str,
    science_card_context: str,
    supporting_chunk_context: str,
) -> str:
    return f"""
You are RAG2, a literature-grounded experiment feasibility advisor.

Your role:
- Review the RAG1 experiment proposal
- Use the available BOM
- Use the retrieved science literature
- Do NOT rewrite the whole experiment
- Do NOT directly replace RAG1
- Give practical advice back to RAG1

Focus only on:
1. Is the experiment feasible with the current BOM?
2. Are there any practical execution limits, uncertainties, or capability constraints within the current BOM?
3. Is the proposal too broad for a first experiment?
4. How should the proposal be narrowed so it becomes easier to execute and interpret?
5. Are the risks listed under RAG1's "Main Risk / Failure Mode" real and important, and if so, how should they be reduced within the current BOM?

Keep the advice practical and simple.
Prioritize executable first-step experiments over ambitious broad studies.
Prefer reducing variables and measurements rather than expanding the design.
Focus on what can realistically be done in the current lab with the listed BOM.
Prefer identifying practical execution limits over abstract evaluation metrics.

Do not give deep theoretical criticism unless it directly affects feasibility or clarity.
Also review the "Main Risk / Failure Mode" section from RAG1.
If any required item, capability, function, or method is not explicitly listed in the BOM, treat it as unavailable.
When an unavailable item is required, do not repair the proposal by adding or assuming it. Instead, revise the proposal so it uses only explicitly listed BOM items.
When reducing risk, prefer simplifying the experiment scope, fixing variables, or using coarser BOM-supported measurements rather than introducing any new instrumentation or monitoring not already in the BOM.
For each major stated risk, judge whether it is a real literature-supported concern or an overstated/secondary concern.
If it is a real concern, explain how RAG1 should reduce or address it without adding new equipment.
If something is optional rather than essential, say that clearly.

Use the Science Literature Cards for overall context.
Use the Supporting Evidence Chunks as the primary evidence for practical advice.
Prefer advice that is directly supported by the supporting chunks.

Do NOT rewrite, shorten, or invent paper titles.
Do NOT cite papers outside the retrieved references.
Use the actual retrieved Science Reference IDs such as SR1, SR2, SR3, etc.
The JSON schema below uses SRx only as a placeholder example.

Do not include any explanation before the JSON.
Your first character must be the opening brace of the JSON object.

The message_to_rag1 should be the practical synthesis of the narrowing_advice and risk_review.
Do not make it generic.
Write it as a concrete revision brief that RAG1 can directly follow.

Status meaning:
- strongly_feasible = executable now with current BOM and only minor assumptions
- mostly_feasible = good core idea, but still needs 1-2 practical revisions
- partially_feasible = plausible, but still too broad or under-specified
- limited = significant execution limits remain
- not_feasible = not realistically executable with current BOM


Available BOM:
{json.dumps(available_bom, indent=2)}

RAG1 Proposal:
{rag1_proposal}

Science Literature Cards:
{science_card_context}

Supporting Evidence Chunks:
{supporting_chunk_context}

Return valid JSON only, in exactly this structure:

{{
  "bom_check": {{
    "status": "strongly_feasible / mostly_feasible / partially_feasible / limited / not_feasible",
    "reason": "short explanation",
    "literature_support": [
      {{
        "reference_id": "SRx",
        "why_it_supports_this": "short explanation"
      }}
    ]
  }},
  "limited_items": [
    {{
      "item": "execution limit 1",
      "why_it_matters": "short explanation of why this limits immediate execution or interpretation",
      "literature_support": [
        {{
          "reference_id": "SRx",
          "why_it_supports_this": "short explanation"
        }}
      ]
    }}
  ],
  "risk_review": [
    {{
      "risk_from_rag1": "risk text from RAG1",
      "is_valid_concern": true,
      "why": "short explanation",
      "suggested_revision": "short practical advice for reducing or addressing the risk within the current BOM",
      "literature_support": [
        {{
          "reference_id": "SRx",
          "why_it_supports_this": "short explanation"
        }}
      ]
    }}
  ],
  "narrowing_advice": [
    {{
      "advice": "advice 1",
      "why": "short explanation",
      "literature_support": [
        {{
          "reference_id": "SRx",
          "why_it_supports_this": "short explanation"
        }}
      ]
    }}
  ],
  "message_to_rag1": "3-4 sentence revision instruction to RAG1. It must briefly incorporate the key narrowing_advice and risk_review items in natural language. It must say what to keep, what to fix, and what to avoid in the next revision."
}}
""".strip()


# =========================================================
# Main callable for orchestrator
# =========================================================
def run_rag2(
    rag1_output: dict,
    cached_sci_evidence: dict | None = None,
    save_output: bool = True,
) -> dict:
    available_bom = rag1_output["available_bom"]
    rag1_proposal = rag1_output["rag1_proposal"]

    # =====================================================
    # ROUND 1: retrieve and cache science evidence
    # =====================================================
    if cached_sci_evidence is None:
        storage_context = StorageContext.from_defaults(
            persist_dir=SCIENCE_STORAGE_DIR
        )
        science_index = load_index_from_storage(storage_context)

        rag2_query = build_rag2_query(available_bom, rag1_proposal)
        retriever = science_index.as_retriever(similarity_top_k=8)
        retrieved_science = retriever.retrieve(rag2_query)

        ranked_science = sorted(
            retrieved_science,
            key=lambda n: overlap_score_text(n.text or "", available_bom),
            reverse=True,
        )

        shortlist = ranked_science[:7]

        print("\nRAG2 round-1 retrieval: caching science evidence pool.")
        print("\nRetrieved science references for RAG2:")
        for i, node in enumerate(shortlist, 1):
            score = overlap_score_text(node.text or "", available_bom)
            print(f"\n[{i}] {node.metadata.get('source_title')}")
            print(f"Card Type: {node.metadata.get('card_type', '')}")
            # print(f"BOM overlap score: {score}")

        labeled_rows = load_labeled_rows(CHUNKS_LABELED_PATH)
        paper_to_chunks = recover_supporting_chunks(
            shortlist=shortlist,
            labeled_rows=labeled_rows,
            bom=available_bom,
            top_chunks_per_paper=3,
        )

        print("Recovered supporting chunks:")
        for node in shortlist:
            source_path = node.metadata.get("source_path", "")
            title = node.metadata.get("source_title", "")
            n = len(paper_to_chunks.get(source_path, []))
            print(f"{title}: {n} chunks")

        cached_sci_evidence = {
            "cards": [
                {
                    "source_title": node.metadata.get("source_title", ""),
                    "doi": node.metadata.get("doi", ""),
                    "source_path": node.metadata.get("source_path", ""),
                    "text": node.text or "",
                    "card_type": node.metadata.get("card_type", ""),
                }
                for node in shortlist
            ],
            "paper_to_chunks": paper_to_chunks,
        }

    # =====================================================
    # ROUND 2+: reuse cached science evidence only
    # =====================================================
    else:
        print("\nRAG2 is reusing cached science evidence pool.")
        shortlist = cached_sci_evidence["cards"]
        paper_to_chunks = cached_sci_evidence["paper_to_chunks"]

    # =====================================================
    # Build science card context
    # =====================================================
    science_blocks = []
    reference_map = {}

    for i, node in enumerate(shortlist, 1):
        if isinstance(node, dict):
            ref_id = f"SR{i}"
            title = node.get("source_title", f"Science Paper {i}")
            doi = node.get("doi", "")
            text = node.get("text", "")
            source_path = node.get("source_path", "")
        else:
            ref_id = f"SR{i}"
            title = node.metadata.get("source_title", f"Science Paper {i}")
            doi = node.metadata.get("doi", "")
            text = node.text or ""
            source_path = node.metadata.get("source_path", "")

        reference_map[ref_id] = {
            "title": title,
            "doi": doi,
            "source_path": source_path,
        }

        science_blocks.append(
            f"""[Science Reference {ref_id}]
Title: {title}
DOI: {doi}

{text}
"""
        )

    science_card_context = "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(science_blocks)

    # =====================================================
    # Build supporting chunk context
    # =====================================================
    chunk_blocks = []
    chunk_idx = 1

    source_path_to_ref_id = {
        meta["source_path"]: ref_id
        for ref_id, meta in reference_map.items()
    }

    for node in shortlist:
        if isinstance(node, dict):
            source_path = node.get("source_path", "")
            title = node.get("source_title", "")
        else:
            source_path = node.metadata.get("source_path", "")
            title = node.metadata.get("source_title", "")

        ref_id = source_path_to_ref_id.get(source_path, "SRx")

        for row in paper_to_chunks.get(source_path, []):
            chunk_blocks.append(
                f"""[Supporting Chunk {chunk_idx}]
Reference ID: {ref_id}
Title: {title}
Chunk ID: {row.get('chunk_id')}
Text:
{row.get('text', '')}
"""
            )
            chunk_idx += 1

    supporting_chunk_context = "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(chunk_blocks)

    # =====================================================
    # Build prompt
    # =====================================================
    rag2_prompt = build_rag2_prompt(
        available_bom=available_bom,
        rag1_proposal=rag1_proposal,
        science_card_context=science_card_context,
        supporting_chunk_context=supporting_chunk_context,
    )

    # =====================================================
    # Generate advice
    # =====================================================
    raw_output = llm.complete(rag2_prompt, max_new_tokens=1600)

    try:
        json_text = extract_json_block(raw_output)
        rag2_advice = json.loads(json_text)
    except Exception as e:
        rag2_advice = {
            "bom_check": {
                "status": "limited",
                "reason": f"Model output could not be parsed as JSON: {str(e)}",
                "literature_support": []
            },
            "limited_items": [
                {
                    "item": "RAG2 structured critique unavailable",
                    "why_it_matters": "The feasibility critique could not be fully structured, so the next revision should stay conservative and BOM-bound.",
                    "literature_support": []
                }
            ],
            "risk_review": [
                {
                    "risk_from_rag1": "Main Risk / Failure Mode could not be reliably reviewed",
                    "is_valid_concern": True,
                    "why": "Because RAG2 parsing failed, the safest assumption is that the stated risks should remain conservative and BOM-bound.",
                    "suggested_revision": "Keep only the most critical execution risks and avoid introducing any new unsupported mitigation step.",
                    "literature_support": []
                }
            ],
            "narrowing_advice": [
                {
                    "advice": "Keep the current experiment core, reduce variables, and avoid adding any new equipment or measurements.",
                    "why": "When RAG2 parsing fails, the safest fallback is to simplify rather than expand.",
                    "literature_support": []
                }
            ],
            "message_to_rag1": (
                "RAG2 parsing failed. Keep the current experiment core, reduce variables, "
                "avoid adding any non-BOM item, and prioritize only the simplest "
                "BOM-supported measurements."
            ),
            "raw_message_to_rag1": raw_output.strip(),
            "parse_failed": True,
        }

    

    print("\n" + "=" * 100)
    print("RAG2 ADVICE")
    print("=" * 100)
    print(json.dumps(rag2_advice, indent=2, ensure_ascii=False))

    if save_output:
        RAG2_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RAG2_OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(rag2_advice, f, indent=2, ensure_ascii=False)
        print(f"\nSaved RAG2 output to: {RAG2_OUTPUT_PATH}")

    print("\nReference Map:")
    for ref_id, meta in reference_map.items():
        print(f"{ref_id}: {meta['title']} | DOI: {meta['doi']}")

    return {
        "rag2_advice": rag2_advice,
        "cached_sci_evidence": cached_sci_evidence,
    }


if __name__ == "__main__":
    raise RuntimeError(
        "scientific_advisor.py is now designed to be called from orchestrator via run_rag2(rag1_output)."
    )
