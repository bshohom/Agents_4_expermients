# Coding-Agent Handoff: High-Level Setup and TODO Map

This document is intentionally verbose. It is meant to give a local coding agent enough context to stand up the repository even if some integrations remain unimplemented.

## 1. What this repo is trying to do

The long-term goal is a **literature-guided experiment proposal system** for constrained scientific planning.

At a high level:

- there is a literature retrieval/setup stage
- there is a persistent embedding/index stage
- there is a runtime orchestration stage
- runtime uses one **local** LLM server on the cluster node
- agent modules retrieve evidence and ask the LLM to reason over that evidence
- the orchestrator decides the order of calls

The core architectural choice for the legacy path is:

- **one model server process**
- **many light clients**
- **one orchestrator controlling workflow**

Do not load the LLM in each agent module.

## 2. Repo intent

This starter repo is deliberately conservative:

- no LangChain required
- no distributed agent messaging required
- no database required for v1
- no runtime embeddings required
- no hosted inference APIs

It is a “single-node serial controller” design.

## 3. Expected runtime behavior

There are now two run modes:

1) **Default mode (`scripts-loop`)**
- root `orchestrator.py` dispatches to `Scripts/orchestrator.py`
- `Scripts` modules run an iterative RAG1 (proposal) <-> RAG2 (advisor) loop
- model loading happens in-process inside `Scripts` modules

2) **Legacy mode (`legacy-simple`)**
- keeps the original single-node served-model pattern described below

Legacy/simple behavior:

When the runtime Slurm job starts:

1. Slurm allocates a GPU node.
2. The job starts a local vLLM server in the background.
3. The Qwen model is loaded **once** into GPU memory.
4. `orchestrator.py` starts.
5. `orchestrator.py` loads the persistent retrieval index.
6. The orchestrator calls retrieval code.
7. The orchestrator formats a prompt with retrieved evidence.
8. The orchestrator sends an HTTP request to the local model server.
9. The orchestrator blocks until the response returns.
10. The orchestrator optionally repeats the process with downstream agents.

That is the whole waiting mechanism for the simple version.

## 4. Why the local server uses an OpenAI-compatible API

This does **not** mean the system uses OpenAI-hosted models.

It means the local server speaks a standard request/response format that is easy to call from Python. vLLM documents this server mode directly, and Qwen documents vLLM as a recommended deployment route. citeturn664162search3turn664162search1

In practice, this lets the code use plain HTTP POST requests to:

- `/v1/chat/completions`
- optionally `/v1/models`

No hosted dependency is required.

## 5. What still needs to be implemented later

This repo includes a functional skeleton, but a coding agent will likely need to improve:

### Setup/data side
- integrate the real Elsevier retrieval pipeline
- convert retrieved full text into normalized paper cards
- split paper backgrounds/discussions for a second RAG space
- define a robust JSON schema for cards
- add better metadata validation

### Runtime side
- add a critique/revision stage
- add structured validation of proposed experiments
- add logging/tracing for every prompt + retrieval call
- add retry policies for malformed model outputs
- add checkpointing of the orchestration state

### Infra side
- pin package versions
- lock tested CUDA / torch / vLLM combinations
- add model-specific chat template handling if needed
- add GPU memory limits / context window tuning

## 6. Expectations for the coding agent

A coding agent working on this repo should:

1. keep the “single served model” assumption intact
2. avoid introducing hidden model loads in agent files
3. keep retrieval and generation separated
4. preserve simple file-based configuration unless complexity is justified
5. document assumptions clearly inside each module

## 7. Current file roles

### `setup/prepare_demo_corpus.py`
Creates a tiny synthetic paper-card corpus so the repo can be tested without Elsevier.

### `setup/build_card_index.py`
Loads JSONL paper cards, creates card text, embeds them with Sentence Transformers, and writes a persistent dense index.

### `agents/rag_cards.py`
Loads the persistent card index and returns top-k retrieved cards for a query.

### `agents/llm_client.py`
Calls the local server over HTTP.

### `agents/proposer.py`
Loads a prompt file, injects BOM + retrieved context, and sends one proposal request to the local LLM.

### `orchestrator.py`
Project entrypoint. Defaults to the `Scripts` multi-agent loop and still supports legacy `retrieve -> propose` mode via CLI.

### `scripts_test_multiprompt.py`
Runs prompt-sweep testing over `prompts/*.txt` using the legacy prompt-based proposer path with timestamped output organization.

### `scripts_test_rag_cards_proposer.py`
Standalone test path to verify the proposer can run without the full orchestrator.

## 8. Design constraints that should not be violated casually

- **No runtime model duplication**.
- **No direct peer-to-peer agent communication for v1**.
- **No runtime embedding generation unless hardware is explicitly allocated for it**.
- **No framework-heavy orchestration unless the current design proves too limiting**.

## 9. Immediate next steps for a coding agent

1. Verify Python imports.
2. Verify `sentence-transformers` model loading on the target machine.
3. Verify `vllm serve` works with a small test model.
4. Build the demo card index.
5. Run `scripts_test_rag_cards_proposer.py`.
6. Run `orchestrator.py`.
7. Replace demo corpus with real paper cards.
8. Add the second RAG space for background/discussion chunks.
9. Add critique and revision agents.

## 10. References for the local-server assumption

- vLLM provides an HTTP server that implements OpenAI-style APIs and can be started with `vllm serve`. citeturn664162search3
- Qwen recommends vLLM for deployment. citeturn664162search1turn664162search7
- Sentence Transformers supports loading models and computing embeddings locally. citeturn664162search5turn664162search8
