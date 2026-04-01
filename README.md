This project investigates whether large language models can propose physically feasible and scientifically meaningful experiments when given a list of physical resources. While recent LLM based research had tested agent’s ability to write code or call simulation tools for research, there are few works testing agent’s ability to propose physical experiments. In many fields of engineering, including additive manufacturing, welding design, and battery cell design, all final outputs rely on physical experimental results over simulation tools, and LLM agents that excels on designing these experiments would help researchers. These experiment design should consider multiple criteria to make it a valuable experiment, such as analytical feasibility, statistical validity, and significancy of the experiment with respect to SOTA academic literature. The central question of this project is whether an agentbased LLM system grounded in a structured bill of materials can generate higher quality experiment proposals than a standard LLM baseline. The proposed method integrates three main components. First, a resource-grounded representation will be constructed by extracting bills of materials (BOM), objectives, and procedural constraints from published experimental papers. This process would consist of RAG based systems, with API pipelines. aiding the access of academic paper retrieval, such as the elsapy python library. Second, we’ll introduce an experiment designing LLM agent that has access to the rag database of relevent papers, with and without access to the generated BOMs. We’ll test multiple agents here, big SOTA LLMs (Claude, Gemini, ChatGPT), and also open source models like Llama, Qwen, etc. We’ll compare their performance against a fine-tuned open source model. Third, structured validator agents will evaluate feasibility, novelty, and statistical soundness using explicit pass or fail criteria. We’ll use a independant big SOTA LLM, with a brief review of the validator’s performance by running through the thinking process and tell if we can trust LLM based systems as validator
## Local environment setup

This project is tested with Python 3.11.

### 1. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install PyTorch first

Install the PyTorch build that matches your machine.

For NVIDIA GPUs with CUDA 12.8 support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

For NVIDIA GPUs with CUDA 12.6 support (Chani):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

For CPU-only:

```bash
pip install torch torchvision torchaudio
```

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

### 4. Register the notebook kernel

```bash
python -m ipykernel install --user --name agents4exp --display-name "Python (agents4exp)"
```

### 5. Set required environment variables

Set your Elsevier API key before running notebooks or scripts:

```bash
export ELSEVIER_APIKEY="your_elsevier_api_key_here"
```

If you want it saved in your terminal:
```bash
echo 'export ELSEVIER_APIKEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc
```

Optional but recommended for faster Hugging Face downloads:

```bash
export HF_TOKEN="your_hf_token_here"
```

### 6. Launch Jupyter

```bash
jupyter lab
```

Then select the `Python (agents4exp)` kernel inside the notebook.

### 7. Important code notes

If you are using the paper-card notebook cell, make sure you have:

```python
from llama_index.core import Document, VectorStoreIndex, Settings
```

and make sure you append normalized cards before writing JSONL:

```python
paper_cards.append(card)
```

---

## Local Ollama setup

This repo uses Ollama for local LLM inference through `llama-index-llms-ollama`.

### 1. Install Ollama on Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Start the Ollama service

```bash
sudo systemctl start ollama
sudo systemctl status ollama
```

If your machine is not using `systemd`, start Ollama manually instead:

```bash
ollama serve
```

### 3. Download a local model

Recommended starter model:

```bash
ollama run qwen3:8b
```

This will download the model the first time and drop you into an interactive session.
Exit with `Ctrl+D` after the download completes.

### 4. Verify the local Ollama server

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3:8b",
  "messages": [{"role": "user", "content": "Hello"}]
}'
```

### 5. Use Ollama in code

```python
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

Settings.llm = Ollama(
    model="qwen3:8b",
    request_timeout=180.0,
    context_window=4096,
)
```

### 6. Make the embedding device explicit

```python
import torch
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core import Settings

Settings.embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
)
```

### 7. Troubleshooting

If Ollama is installed but your notebook still raises `ConnectionError: Failed to connect to Ollama`, check:

```bash
ollama --version
ollama list
curl http://localhost:11434/api/tags
sudo systemctl status ollama
```

If the service is running and the model is present, rerun the notebook with:

```python
Settings.llm = Ollama(model="qwen3:8b", request_timeout=180.0, context_window=4096)
```

---

## Cluster / Slurm setup

The recommended cluster workflow is:

1. Build a Python environment once on a compatible machine.
2. Install the correct PyTorch build for the target cluster.
3. Install the repo dependencies.
4. Package the environment.
5. Unpack it inside the Slurm job.
6. Run a Python script, not a notebook.

### Why this is the recommended cluster pattern

For batch jobs, it is usually easier and more reproducible to:
- prepackage the environment
- unpack it into node-local scratch
- run a script like `python run_pipeline.py ...`

This avoids relying on the cluster's base Python and avoids ad hoc notebook state.

### Recommended cluster project structure

```text
Agents_4_expermients/
├── requirements.txt
├── submit.sh
├── envs/
│   └── agents4exp.tar.gz
├── scripts/
│   ├── search_and_filter.py
│   ├── build_paper_cards.py
│   └── propose_experiment.py
├── outputs/
├── logs/
└── selected_relevant_fulltext_papers_1000.xlsx
```

### Build and package the environment

If you use conda/mamba, one good approach is:

```bash
mamba create -n agents4exp python=3.11 -y
mamba activate agents4exp

# Pick ONE PyTorch install
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# or:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
mamba install -n agents4exp conda-pack -y

mkdir -p envs
conda-pack -n agents4exp -o envs/agents4exp.tar.gz
```

If you prefer `venv`, you can also tar the environment manually, but `conda-pack` is generally cleaner for cluster relocation.

### Environment variables on cluster

Your scripts should read secrets from environment variables, not hardcoded strings.

Example:

```python
import os
apikey = os.environ["ELSEVIER_APIKEY"]
```

You can export these before `sbatch` or inside your cluster job wrapper.

### Recommended cluster runtime behavior

- unpack the environment into `$SLURM_TMPDIR`
- set Hugging Face caches to scratch
- write job outputs to `outputs/$SLURM_JOB_ID`
- keep logs in `logs/`

---

## Example `submit.sh` for a prepackaged environment

```bash
#!/bin/bash
#SBATCH --job-name=agents4exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

echo "Job started on $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"

PROJECT_ROOT="$HOME/src/Agents_4_expermients"
PACKED_ENV="$PROJECT_ROOT/envs/agents4exp.tar.gz"
ENV_DIR="${SLURM_TMPDIR}/agents4exp_env"

mkdir -p "$ENV_DIR"
tar -xzf "$PACKED_ENV" -C "$ENV_DIR"

source "${ENV_DIR}/bin/activate"

if [ -f "${ENV_DIR}/bin/conda-unpack" ]; then
    conda-unpack
fi

export HF_HOME="${SLURM_TMPDIR}/hf_home"
export TRANSFORMERS_CACHE="${SLURM_TMPDIR}/hf_home/transformers"
export XDG_CACHE_HOME="${SLURM_TMPDIR}/xdg_cache"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

export ELSEVIER_APIKEY="${ELSEVIER_APIKEY:?ELSEVIER_APIKEY not set}"
export HF_TOKEN="${HF_TOKEN:-}"

cd "$PROJECT_ROOT"

python scripts/build_paper_cards.py \
    --input selected_relevant_fulltext_papers_1000.xlsx \
    --outdir outputs/${SLURM_JOB_ID}
```

---

## Optional: cluster job using Ollama inside the allocation

This is possible, but usually more fragile than the plain packaged-environment approach.
Only use this if you want to keep the local Ollama-backed path unchanged on the cluster.

### Example `submit_ollama.sh`

```bash
#!/bin/bash
#SBATCH --job-name=agents4exp-ollama
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="$HOME/src/Agents_4_expermients"
PACKED_ENV="$PROJECT_ROOT/envs/agents4exp.tar.gz"
ENV_DIR="${SLURM_TMPDIR}/agents4exp_env"

mkdir -p "$ENV_DIR"
tar -xzf "$PACKED_ENV" -C "$ENV_DIR"

source "${ENV_DIR}/bin/activate"

if [ -f "${ENV_DIR}/bin/conda-unpack" ]; then
    conda-unpack
fi

export HF_HOME="${SLURM_TMPDIR}/hf_home"
export TRANSFORMERS_CACHE="${SLURM_TMPDIR}/hf_home/transformers"
export XDG_CACHE_HOME="${SLURM_TMPDIR}/xdg_cache"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

export ELSEVIER_APIKEY="${ELSEVIER_APIKEY:?ELSEVIER_APIKEY not set}"
export HF_TOKEN="${HF_TOKEN:-}"

export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_MODELS="${SLURM_TMPDIR}/ollama_models"
mkdir -p "$OLLAMA_MODELS"

ollama serve > "${SLURM_TMPDIR}/ollama.log" 2>&1 &
OLLAMA_PID=$!

sleep 5

ollama run qwen3:8b <<< "hello" || true

cd "$PROJECT_ROOT"

python scripts/propose_experiment.py \
    --input selected_relevant_fulltext_papers_1000.xlsx \
    --outdir outputs/${SLURM_JOB_ID}

kill "$OLLAMA_PID" || true
```

### When to use the Ollama-in-job approach

Use it only if:
- you want to preserve the same LlamaIndex + Ollama code path on the cluster
- the cluster allows background processes on compute nodes
- the model can be downloaded ahead of time or cached on shared storage

Prefer the plain packaged-environment approach if:
- the cluster blocks outbound internet from compute nodes
- you want simple, reproducible batch jobs
- you are willing to run a non-Ollama backend for cluster execution

---

## Suggested next refactor for this repo

For the cleanest local + cluster story, move notebook logic into scripts such as:

```text
scripts/
├── search_and_filter.py
├── build_paper_cards.py
└── propose_experiment.py
```

Recommended responsibilities:

- `search_and_filter.py`
  - fetch papers
  - read full text
  - score chunks
  - save filtered chunk evidence

- `build_paper_cards.py`
  - group chunks by paper
  - call the LLM for one structured card per paper
  - normalize card schema
  - save JSONL
  - build vector index

- `propose_experiment.py`
  - load the paper-card index
  - apply BOM constraints
  - retrieve relevant cards
  - generate one BOM-feasible experiment proposal

---

## Recommended local install order

1. Create the Python environment.
2. Install the correct PyTorch build.
3. `pip install -r requirements.txt`
4. Install and start Ollama.
5. Download `qwen3:8b` if you have min 10GD ram, else `qwen3:3b`
6. Launch Jupyter or run the scripts directly.

---

## Minimal `requirements.txt`

If you want the matching `requirements.txt`, use:

```text
pandas
openpyxl
elsapy==0.5.1

llama-index-core
llama-index-embeddings-langchain
llama-index-llms-ollama

langchain-huggingface
sentence-transformers
transformers
huggingface-hub

jupyter
ipykernel
```

Notes:
- Keep `torch`, `torchvision`, and `torchaudio` out of `requirements.txt`
- Install them separately so each machine can choose `cu128`, `cu126`, or CPU-only