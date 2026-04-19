#!/bin/bash
#SBATCH --job-name=battery-temp-sweep
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-79
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
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

# Change these only if you want a different local model / embedding.
# export LOCAL_LLM_MODEL="Qwen/Qwen3-8B"
# export LOCAL_EMBED_MODEL="BAAI/bge-small-en-v1.5"

PROJECT_ROOT="/trace/group/forgelab/sboseban/code/Agents_4_Expermients"

PROMPT_FILE="${PROJECT_ROOT}/prompts/battery_p2_fixed.jsonl"
REVIEWER_FILE="${PROJECT_ROOT}/prompts/battery_r4_fixed.jsonl"
BOM_FILE="${PROJECT_ROOT}/prompts/battery_boms_10.jsonl"

# Point these at the corpus you want to use.
CORPUS_ROOT="${PROJECT_ROOT}/external_corpora/chani/outputs_merged"
EXP_INDEX_DIR="${CORPUS_ROOT}/paper_memory_storage_experiment"
SCI_INDEX_DIR="${CORPUS_ROOT}/paper_memory_storage_science"
CHUNKS_LABELED_PATH="${CORPUS_ROOT}/chunks_labeled.jsonl"

OUTPUT_ROOT="${PROJECT_ROOT}/outputs_battery_temp_sweep"

python "${PROJECT_ROOT}/Scripts/orchestrator.py" \
  --proposal-prompts-file "${PROMPT_FILE}" \
  --reviewer-prompts-file "${REVIEWER_FILE}" \
  --bom-file "${BOM_FILE}" \
  --experiment-index-dir "${EXP_INDEX_DIR}" \
  --science-index-dir "${SCI_INDEX_DIR}" \
  --chunks-labeled-path "${CHUNKS_LABELED_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --proposal-temperatures "0.0,0.2,0.5,0.8" \
  --reviewer-temperature 0.0 \
  --seed-values "0,1" \
  --top-p 0.95 \
  --max-rounds 5
