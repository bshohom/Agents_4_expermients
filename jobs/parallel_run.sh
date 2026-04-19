#!/bin/bash
#SBATCH --job-name=paper-pipeline-array
#SBATCH --partition=batch
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=024:20:00
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

PROJECT_ROOT="/trace/group/forgelab/sboseban/code/Agents_4_Expermients"
SHARD_ROOT="${PROJECT_ROOT}/lit_shards"

SHARD_ID="${SLURM_ARRAY_TASK_ID}"
echo "Running shard ${SHARD_ID}..."
SHARD_LIT_DIR="${SHARD_ROOT}/shard_${SHARD_ID}"
EMBED_OUT="${PROJECT_ROOT}/outputs_parallel/embed/shard_${SHARD_ID}"
LABEL_OUT="${PROJECT_ROOT}/outputs_parallel/label/shard_${SHARD_ID}"
CARD_OUT="${PROJECT_ROOT}/outputs_parallel/cards/shard_${SHARD_ID}"

mkdir -p "${EMBED_OUT}" "${LABEL_OUT}" "${CARD_OUT}"

python "${PROJECT_ROOT}/Scripts/embed.py" \
  --lit-dir "${SHARD_LIT_DIR}" \
  --out-dir "${EMBED_OUT}" \
  --max-seconds 7800

if [[ -f "${EMBED_OUT}/chunks.jsonl" ]]; then
  python "${PROJECT_ROOT}/Scripts/cards_divider.py" \
    --chunks-path "${EMBED_OUT}/chunks.jsonl" \
    --out-dir "${LABEL_OUT}" \
    --max-seconds 7800
fi

if [[ -f "${LABEL_OUT}/chunks_labeled.jsonl" ]]; then
  python "${PROJECT_ROOT}/Scripts/paper_card_only.py" \
    --out-dir "${CARD_OUT}" \
    --chunks-labeled-path "${LABEL_OUT}/chunks_labeled.jsonl" \
    --cards-path "${CARD_OUT}/paper_memory_cards.jsonl" \
    --max-seconds 7800
fi