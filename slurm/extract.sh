#!/bin/bash
#SBATCH --job-name=probe_extract
#SBATCH --partition=mlgroup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-79%8
#SBATCH --output=logs/probe_extract_%A_%a.out
#SBATCH --error=logs/probe_extract_%A_%a.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate navel
cd ~/wordle

CONDITIONS=(flat_lm flat_random hrl_lm hrl_random)
ARCHS=(flat flat hrl hrl)

COND_IDX=$((SLURM_ARRAY_TASK_ID / 20))
ITER_IDX=$((SLURM_ARRAY_TASK_ID % 20))

CONDITION=${CONDITIONS[$COND_IDX]}
ARCH=${ARCHS[$COND_IDX]}
ITER=$(((ITER_IDX + 1) * 5))
ITER_PAD=$(printf "%04d" $ITER)

EXP="exp_0.0001_0.01_5_0.0_0.0_0.0_10.0_6_8.0"
CHECKPOINT="$PWD/checkpoints/$CONDITION/seed_42/$EXP/iter_$ITER_PAD"
NAME="$CONDITION/iter_$ITER_PAD"

echo "============================================"
echo "Job       : $SLURM_JOB_ID"
echo "Task      : $SLURM_ARRAY_TASK_ID"
echo "Node      : $(hostname)"
echo "Condition : $CONDITION"
echo "Arch      : $ARCH"
echo "Iteration : $ITER"
echo "Checkpoint: $CHECKPOINT"
echo "============================================"

if [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    exit 1
fi

python -m probing.extract_representations \
    --checkpoint "$CHECKPOINT" \
    --architecture "$ARCH" \
    --name "$NAME"
