#!/bin/bash
#SBATCH --job-name=lit-agent-run
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/trace/group/forgelab/sboseban/code/Agents_4_Expermients/logs/lit_agent_%j.out
#SBATCH --error=/trace/group/forgelab/sboseban/code/Agents_4_Expermients/logs/lit_agent_%j.err

set -euo pipefail

module load cuda/12.8
module load gcc/11.3.0
module load anaconda3/2023.03-1

eval "$(conda shell.bash hook)"
conda activate /trace/group/forgelab/sboseban/envs/agents4experimentsa

export HF_HOME=/trace/group/forgelab/sboseban/hf_cache
export TRANSFORMERS_CACHE=/trace/group/forgelab/sboseban/hf_cache
export HUGGINGFACE_HUB_CACHE=/trace/group/forgelab/sboseban/hf_cache

PROJECT_ROOT="/trace/group/forgelab/sboseban/code/Agents_4_Expermients"
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

# Main workflow: run Scripts-based multi-agent loop via root orchestrator entrypoint.
python orchestrator.py \
  --mode scripts-loop \
  --output_file outputs/slurm_scripts_loop_output.json
