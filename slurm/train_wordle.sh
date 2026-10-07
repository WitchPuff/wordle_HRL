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
#   sbatch slurm/train_wordle.sh flat lm 42
#   sbatch slurm/train_wordle.sh hrl lm 42
#   sbatch slurm/train_wordle.sh flat random 42
#   sbatch slurm/train_wordle.sh hrl random 42
# ============================================================

ARCHITECTURE=${1:-hrl}
PRIOR=${2:-lm}
SOFT_MASK_PENALTY=${3:-6.0}

ITERATIONS=${4:-100}
CHECKPOINT_EVERY=${5:-5}
EVAL_EVERY=${6:-1}

LR=${7:-1e-4}
ENTROPY_COEFF=${8:-0.01}
EPOCHS=${9:-5}

STEP_PENALTY=${10:--0.1}
YELLOW_REWARD=${11:-0.0}
GREEN_REWARD=${12:-0.0}
INFO_GAIN_WEIGHT=${13:-0.0}
SOLVE_REWARD=${14:-10.0}
MAX_GUESSES=${15:-6}
SEED=${16:-42}


# ============================================================
# Validate
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

mkdir -p logs results checkpoints


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
echo "Evaluation   : every ${EVAL_EVERY} iterations"
echo "LR           : ${LR}"
echo "Entropy      : ${ENTROPY_COEFF}"
echo "Epochs       : ${EPOCHS}"
echo "Max guesses  : ${MAX_GUESSES}"
echo "Soft penalty : ${SOFT_MASK_PENALTY}"
echo "Start        : $(date)"
echo "============================================================"

echo
echo "GPU:"
nvidia-smi
echo


# ============================================================
# Training
# ============================================================

python -u -m training.train \
    --architecture "$ARCHITECTURE" \
    --prior "$PRIOR" \
    --seed "$SEED" \
    --iterations "$ITERATIONS" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --eval-every "$EVAL_EVERY" \
    --lr "$LR" \
    --entropy-coeff "$ENTROPY_COEFF" \
    --num-epochs "$EPOCHS" \
    --step-penalty "$STEP_PENALTY" \
    --yellow-reward "$YELLOW_REWARD" \
    --green-reward "$GREEN_REWARD" \
    --info-gain-weight "$INFO_GAIN_WEIGHT" \
    --solve-reward "$SOLVE_REWARD" \
    --max-guesses "$MAX_GUESSES" \
    --soft-mask-penalty "$SOFT_MASK_PENALTY"


# ============================================================
# Finished
# ============================================================

echo
echo "============================================================"
echo "Training finished"
echo "End: $(date)"
echo "============================================================"