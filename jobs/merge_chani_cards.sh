#!/bin/bash
#SBATCH --job-name=chani-merge-cards
#SBATCH --partition=batch
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:30:00
#SBATCH --output=/trace/group/forgelab/sboseban/code/Agents_4_Expermients/logs/%x_%A_%a.out
#SBATCH --error=/trace/group/forgelab/sboseban/code/Agents_4_Expermients/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/trace/group/forgelab/sboseban/code/Agents_4_Expermients"
CORPUS_DIR="${PROJECT_ROOT}/external_corpora/chani"
MERGED_OUT="${CORPUS_DIR}/outputs_merged"

module load gcc/11.3.0
module load anaconda3/2023.03-1

eval "$(conda shell.bash hook)"
conda activate /trace/group/forgelab/sboseban/envs/agents4experiments

export HF_HOME=/trace/group/forgelab/sboseban/hf_cache
export TRANSFORMERS_CACHE=/trace/group/forgelab/sboseban/hf_cache
export HUGGINGFACE_HUB_CACHE=/trace/group/forgelab/sboseban/hf_cache
export LOCAL_EMBED_MODEL="BAAI/bge-small-en-v1.5"

mkdir -p "${MERGED_OUT}"

find "${CORPUS_DIR}/outputs_parallel/embed" -name "chunks.jsonl" -exec cat {} \; \
  > "${MERGED_OUT}/chunks.jsonl"

find "${CORPUS_DIR}/outputs_parallel/label" -name "chunks_labeled.jsonl" -exec cat {} \; \
  > "${MERGED_OUT}/chunks_labeled.jsonl"

find "${CORPUS_DIR}/outputs_parallel/cards" -name "paper_memory_cards.jsonl" -exec cat {} \; \
  > "${MERGED_OUT}/paper_memory_cards.jsonl"

echo "Merged files created:"
wc -l "${MERGED_OUT}/chunks.jsonl"
wc -l "${MERGED_OUT}/chunks_labeled.jsonl"
wc -l "${MERGED_OUT}/paper_memory_cards.jsonl"

python "${PROJECT_ROOT}/Scripts/build_indexes_from_cards.py" \
  --cards-path "${MERGED_OUT}/paper_memory_cards.jsonl" \
  --out-dir "${MERGED_OUT}"

echo
echo "Done."
find "${MERGED_OUT}" -maxdepth 2 -type f | sort