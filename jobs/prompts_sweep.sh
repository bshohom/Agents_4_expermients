#!/bin/bash
#SBATCH --job-name=prompt-sweep
#SBATCH --partition=batch
#SBATCH --array=0-99
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
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

PROPOSAL_PROMPTS_FILE="${PROJECT_ROOT}/prompts/test_proposer_prompts.jsonl"
REVIEWER_PROMPTS_FILE="${PROJECT_ROOT}/prompts/test_reviewer_prompts.jsonl"
BOM_FILE="${PROJECT_ROOT}/prompts/test_boms.jsonl"

EXP_INDEX_DIR="${PROJECT_ROOT}/outputs_parallel/cards/merged/paper_memory_storage_experiment"
SCI_INDEX_DIR="${PROJECT_ROOT}/outputs_parallel/cards/merged/paper_memory_storage_science"

CHUNKS_LABELED_PATH="${PROJECT_ROOT}/outputs_parallel/label/merged/chunks_labeled.jsonl"
# --experiment-index-dir /trace/group/forgelab/sboseban/code/Agents_4_Expermients/outputs_parallel/cards/merged/paper_memory_storage_experiment
# --science-index-dir /trace/group/forgelab/sboseban/code/Agents_4_Expermients/outputs_parallel/cards/merged/paper_memory_storage_science
# --chunks-labeled-path /trace/group/forgelab/sboseban/code/Agents_4_Expermients/outputs_parallel/label/merged/chunks_labeled.jsonl

OUTPUT_ROOT="${PROJECT_ROOT}/outputs_prompt_sweep"

TASK_ID="${SLURM_ARRAY_TASK_ID}"

python "${PROJECT_ROOT}/Scripts/orchestrator.py" \
  --proposal-prompts-file "${PROPOSAL_PROMPTS_FILE}" \
  --reviewer-prompts-file "${REVIEWER_PROMPTS_FILE}" \
  --bom-file "${BOM_FILE}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --experiment-index-dir "${EXP_INDEX_DIR}" \
  --science-index-dir "${SCI_INDEX_DIR}" \
  --chunks-labeled-path "${CHUNKS_LABELED_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --max-rounds 5