#!/bin/bash
#SBATCH --job-name=chani-cards
#SBATCH --partition=batch
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:30:00
#SBATCH --output=/trace/group/forgelab/sboseban/code/Agents_4_Expermients/logs/%x_%A_%a.out
#SBATCH --error=/trace/group/forgelab/sboseban/code/Agents_4_Expermients/logs/%x_%A_%a.err

set -euo pipefail

module load gcc/11.3.0
module load anaconda3/2023.03-1

eval "$(conda shell.bash hook)"
conda activate /trace/group/forgelab/sboseban/envs/agents4experiments

export HF_HOME=/trace/group/forgelab/sboseban/hf_cache
export TRANSFORMERS_CACHE=/trace/group/forgelab/sboseban/hf_cache
export HUGGINGFACE_HUB_CACHE=/trace/group/forgelab/sboseban/hf_cache
export LOCAL_EMBED_MODEL="BAAI/bge-small-en-v1.5"

PROJECT_ROOT="/trace/group/forgelab/sboseban/code/Agents_4_Expermients"
CORPUS_DIR="${PROJECT_ROOT}/external_corpora/chani"

SHARD_ID="${SLURM_ARRAY_TASK_ID}"
SHARD_LIT_DIR="${CORPUS_DIR}/lit_shards/shard_${SHARD_ID}"

EMBED_OUT="${CORPUS_DIR}/outputs_parallel/embed/shard_${SHARD_ID}"
LABEL_OUT="${CORPUS_DIR}/outputs_parallel/label/shard_${SHARD_ID}"
CARD_OUT="${CORPUS_DIR}/outputs_parallel/cards/shard_${SHARD_ID}"

mkdir -p "${EMBED_OUT}" "${LABEL_OUT}" "${CARD_OUT}" "${PROJECT_ROOT}/logs"

echo "============================================================"
echo "Starting shard ${SHARD_ID}"
echo "Node: $(hostname)"
echo "Shard lit dir: ${SHARD_LIT_DIR}"
echo "============================================================"

if [[ ! -d "${SHARD_LIT_DIR}" ]]; then
  echo "Missing shard dir: ${SHARD_LIT_DIR}"
  exit 1
fi

echo
echo "[1/3] embed.py"
python "${PROJECT_ROOT}/Scripts/embed.py" \
  --lit-dir "${SHARD_LIT_DIR}" \
  --out-dir "${EMBED_OUT}" \
  --max-seconds 3600

if [[ ! -f "${EMBED_OUT}/chunks.jsonl" ]]; then
  echo "embed.py did not produce ${EMBED_OUT}/chunks.jsonl"
  exit 1
fi

echo
echo "[2/3] cards_divider.py"
python "${PROJECT_ROOT}/Scripts/cards_divider.py" \
  --chunks-path "${EMBED_OUT}/chunks.jsonl" \
  --out-dir "${LABEL_OUT}" \
  --max-seconds 3600

if [[ ! -f "${LABEL_OUT}/chunks_labeled.jsonl" ]]; then
  echo "cards_divider.py did not produce ${LABEL_OUT}/chunks_labeled.jsonl"
  exit 1
fi

echo
echo "[3/3] paper_card_only.py"
python "${PROJECT_ROOT}/Scripts/paper_card_only.py" \
  --out-dir "${CARD_OUT}" \
  --chunks-labeled-path "${LABEL_OUT}/chunks_labeled.jsonl" \
  --cards-path "${CARD_OUT}/paper_memory_cards.jsonl" \
  --max-seconds 3600

if [[ ! -f "${CARD_OUT}/paper_memory_cards.jsonl" ]]; then
  echo "paper_card_only.py did not produce ${CARD_OUT}/paper_memory_cards.jsonl"
  exit 1
fi

echo
echo "Shard ${SHARD_ID} complete."
wc -l "${EMBED_OUT}/chunks.jsonl" || true
wc -l "${LABEL_OUT}/chunks_labeled.jsonl" || true
wc -l "${CARD_OUT}/paper_memory_cards.jsonl" || true