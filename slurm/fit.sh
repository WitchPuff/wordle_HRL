#!/bin/bash
#SBATCH --job-name=fit_probes
#SBATCH --partition=mlgroup
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fit_probes_%j.out
#SBATCH --error=logs/fit_probes_%j.err

source /home/students/yhe/anaconda3/etc/profile.d/conda.sh
conda activate navel
cd /home/students/yhe/wordle

python -m probing.fit_probes