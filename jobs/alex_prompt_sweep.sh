#!/bin/bash
#SBATCH --job-name=alex-prompt-sweep
#SBATCH --partition=batch
#SBATCH --array=0-124
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
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
CORPUS_DIR="${PROJECT_ROOT}/external_corpora/alex"

python "${PROJECT_ROOT}/Scripts/orchestrator.py" \
  --experiment-index-dir "${CORPUS_DIR}/paper_memory_storage_experiment" \
  --science-index-dir "${CORPUS_DIR}/paper_memory_storage_science" \
  --chunks-labeled-path "${CORPUS_DIR}/chunks_labeled.jsonl" \
  --proposal-prompts-file "${PROJECT_ROOT}/prompts/welding_proposer_prompts.jsonl" \
  --reviewer-prompts-file "${PROJECT_ROOT}/prompts/test_reviewer_prompts.jsonl" \
  --bom-file "${PROJECT_ROOT}/prompts/welding_boms.jsonl" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${CORPUS_DIR}/outputs_prompt_sweep" \
  --max-rounds 5