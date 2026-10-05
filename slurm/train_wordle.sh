#!/bin/bash
#SBATCH --job-name=wordle
#SBATCH --partition=mlgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/wordle_%j.out
#SBATCH --error=logs/wordle_%j.err

set -euo pipefail


# ============================================================
# Arguments
#
# Usage:
#
#   sbatch slurm/train_wordle.sh flat lm 42
#   sbatch slurm/train_wordle.sh hrl  lm 42
#
# ============================================================

ARCHITECTURE=${1:-flat}
PRIOR=${2:-lm}
SEED=${3:-42}

ITERATIONS=${4:-100}
CHECKPOINT_EVERY=${5:-5}


# ============================================================
# Validate arguments
# ============================================================

if [[ "$ARCHITECTURE" != "flat" && "$ARCHITECTURE" != "hrl" ]]; then
    echo "ERROR: architecture must be 'flat' or 'hrl'"
    exit 1
fi

if [[ "$PRIOR" != "lm" && "$PRIOR" != "random" ]]; then
    echo "ERROR: prior must be 'lm' or 'random'"
    exit 1
fi


# ============================================================
# Environment
# ============================================================

source ~/anaconda3/etc/profile.d/conda.sh
conda activate navel

cd ~/wordle

mkdir -p logs
mkdir -p results
mkdir -p checkpoints


# ============================================================
# Job information
# ============================================================

echo "============================================================"
echo "Wordle RL training"
echo "============================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Architecture : ${ARCHITECTURE}"
echo "Prior        : ${PRIOR}"
echo "Seed         : ${SEED}"
echo "Iterations   : ${ITERATIONS}"
echo "Checkpoint   : every ${CHECKPOINT_EVERY} iterations"
echo "Start        : $(date)"
echo "============================================================"

echo
echo "GPU:"
nvidia-smi
echo


# ============================================================
# Training
# ============================================================

if [[ "$ARCHITECTURE" == "flat" ]]; then

    python -u -m training.train_rl \
        --seed "$SEED" \
        --prior "$PRIOR" \
        --iterations "$ITERATIONS" \
        --checkpoint-every "$CHECKPOINT_EVERY"

elif [[ "$ARCHITECTURE" == "hrl" ]]; then

    python -u -m training.train_hrl \
        --seed "$SEED" \
        --prior "$PRIOR" \
        --num-options 3 \
        --iterations "$ITERATIONS" \
        --checkpoint-every "$CHECKPOINT_EVERY"

fi


# ============================================================
# Finished
# ============================================================

echo
echo "============================================================"
echo "Training finished"
echo "End: $(date)"
echo "============================================================"