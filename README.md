# Literature Agent Template (Local Models on a Cluster)

This is a minimal starter repo for a **single-node, single-LLM, multi-agent-by-orchestration** workflow.

The intended flow is:

1. **Setup stage**
   - retrieve / clean literature
   - convert papers into normalized paper cards
   - embed the cards
   - save a persistent retrieval index

2. **Run stage**
  - start one **local** model server on the GPU node (for legacy/simple mode)
  - run `orchestrator.py`
  - default runtime uses the Scripts multi-agent loop (RAG1 proposal + RAG2 critique/revision)
  - legacy mode still supports prompt-template-driven retrieve -> propose runs

Nothing in this template uses a hosted OpenAI model.
The local server simply exposes an **OpenAI-compatible HTTP interface**, which vLLM supports for locally served models. vLLM documents an HTTP server that implements OpenAI-style chat/completions APIs, and Qwen's docs recommend vLLM for deployment. citeturn664162search3turn664162search1

## What is included

- `agents/llm_client.py` — thin HTTP client for a local vLLM server
- `agents/rag_cards.py` — lightweight embedding-based retrieval over paper cards
- `agents/proposer.py` — prompt-driven proposer agent
- `setup/build_card_index.py` — builds a simple persistent index from JSONL paper cards
- `setup/prepare_demo_corpus.py` — creates a small demo paper-card corpus for testing
- `orchestrator.py` — serial controller for proposal generation
- `jobs/submit_run.sh` — Slurm template that starts the local model server and then the orchestrator
- `jobs/start_vllm_local.sh` — helper for local interactive testing on a GPU machine
- `scripts_test_rag_cards_proposer.py` — direct test for `rag_cards + proposer`
- `AGENT_HANDOFF.md` — verbose setup/context notes for a coding agent

## Minimal install sketch

This repo assumes a Python environment with at least:

- `requests`
- `numpy`
- `sentence-transformers`
- `torch`
- optionally `vllm` for local serving on GPU nodes

Sentence Transformers supports loading local or hub-hosted embedding models and using them to compute embeddings. citeturn664162search5turn664162search8

## Suggested first test

1. Build the demo corpus:

```bash
python setup/prepare_demo_corpus.py
python setup/build_card_index.py \
  --input_jsonl data/demo/paper_cards_demo.jsonl \
  --output_dir indices/paper_cards_demo
```

2. Start a local model server on a GPU machine:

```bash
bash jobs/start_vllm_local.sh Qwen/Qwen3-8B
```

3. Run the standalone retrieval + proposer test:

```bash
python scripts_test_rag_cards_proposer.py \
  --index_dir indices/paper_cards_demo \
  --prompt_file prompts/propose_experiment.txt \
  --bom_file data/demo/bom_demo.json
```

4. Run the default orchestrator (Scripts multi-agent loop):

```bash
python orchestrator.py \
  --mode scripts-loop \
  --output_file outputs/demo_scripts_loop_output.json
```

5. Run the legacy simple orchestrator (prompt-template path):

```bash
python orchestrator.py \
  --mode legacy-simple \
  --index_dir indices/paper_cards_demo \
  --prompt_file prompts/propose_experiment.txt \
  --bom_file data/demo/bom_demo.json \
  --output_file outputs/demo_orchestrator_legacy_output.json
```

## Command guide

### A) Default project workflow (Scripts agent loop)

Use this as the main project run path.

```bash
python orchestrator.py \
  --mode scripts-loop \
  --output_file outputs/scripts_loop_output.json
```

Notes:
- This calls `Scripts/orchestrator.py` from the root entrypoint.
- `Scripts/` files are kept unchanged as reference implementations.

### B) Legacy prompt-based single-pass workflow

Use this to keep compatibility with prompt-template testing.

```bash
python orchestrator.py \
  --mode legacy-simple \
  --index_dir indices/paper_cards_demo \
  --prompt_file prompts/propose_experiment.txt \
  --bom_file data/demo/bom_demo.json \
  --output_file outputs/legacy_simple_output.json
```

### C) Multi-prompt testing with organized outputs

Runs all prompt files matching a glob and stores one JSON per prompt plus a summary.

```bash
python scripts_test_multiprompt.py \
  --index_dir indices/paper_cards_demo \
  --bom_file data/demo/bom_demo.json \
  --prompt_glob 'prompts/*.txt' \
  --output_dir outputs/multiprompt
```

Outputs are organized under timestamped folders like:
- `outputs/multiprompt/<UTC_TIMESTAMP>/<prompt_name>.json`
- `outputs/multiprompt/<UTC_TIMESTAMP>/summary.json`

## Notes

This template intentionally keeps retrieval and generation simple:

- Retrieval is dense cosine similarity over persisted embeddings.
- Orchestration is serial and deterministic.
- There is exactly one LLM server process.
- Requests “wait” naturally because the orchestrator makes blocking HTTP calls; if you later add concurrency, the server still has a request queue. vLLM exposes an OpenAI-compatible server and queue/scheduling machinery rather than requiring each client to load its own model. citeturn664162search3turn664162search21
