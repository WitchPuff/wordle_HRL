# Wordle RL and Hierarchical Representation Learning

This repository compares Flat RL and hierarchical RL (HRL) agents for Wordle under two word-representation conditions: pretrained language-model (LM) embeddings and random embeddings.

The main experiment follows a 2 × 2 design:

| Architecture | Word representation |
|---|---|
| Flat | LM |
| Flat | Random |
| HRL | LM |
| HRL | Random |

All four conditions use the same Wordle environment, data splits, reward function, candidate filtering, soft candidate bias, and PPO training configuration.

## Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate wd
```

Run all commands from the repository root.

## 1. Prepare Data and Embeddings

Prepare the Wordle vocabulary, train/validation/test splits, and word embeddings:

```bash
python -m training.prepare
```

This creates the artifacts required by the training experiments, including the dataset splits and LM/random word representations.

## 2. Run the Four Training Experiments

Submit the four experimental conditions:

```bash
sbatch slurm/train_wordle.sh flat lm
sbatch slurm/train_wordle.sh hrl lm
sbatch slurm/train_wordle.sh flat random
sbatch slurm/train_wordle.sh hrl random
```

The two experimental factors are:

- `flat` / `hrl`: policy architecture
- `lm` / `random`: word representation prior

Training checkpoints and metrics are saved under `results/`.

You can monitor the submitted jobs with:

```bash
squeue -u $USER
```

Wait until all four training jobs have completed before running the subsequent analysis.

## 3. Plot Training Results

Generate the training-performance figure:

```bash
python -m training.plot
```

The resulting figure compares solve rate and mean number of guesses among solved episodes for all four experimental conditions.

## 4. Generate Fixed Probing States

Generate a fixed set of validation and test states for representation probing:

```bash
python -m probing.generate_fixed_states
```

The same states are used across architectures, representation priors, and training checkpoints to make probe results directly comparable.

The probing concepts include task-state variables such as candidate-set uncertainty, timestep, green information, and position-specific constraints.

## 5. Extract Internal Representations

Extract representations from the saved checkpoints:

```bash
sbatch slurm/extract.sh
```

For the Flat architecture, the hidden state representation is extracted.

For HRL, three representations are extracted:

```text
high-level representation
strategy representation
low-level representation
```

Wait until the extraction jobs have completed before fitting the probes.

## 6. Fit Linear Probes

Submit the probe-fitting job:

```bash
sbatch slurm/fit.sh
```

Continuous concepts are evaluated using Ridge regression and test $begin:math:text$R\^2$end:math:text$. Binary position-specific concepts are evaluated using logistic regression and balanced accuracy.

Probe model selection is performed on validation states, with trajectories grouped by target word to prevent states from the same Wordle trajectory from leaking across cross-validation folds. Final scores are evaluated on held-out test states.

## 7. Plot Probing Results

After probe fitting has completed, generate the probing figure:

```bash
python -m probing.plot
```

The figure compares concept decodability across:

- Flat LM vs. Flat Random
- HRL LM vs. HRL Random
- high-level, strategy, and low-level HRL representations
- training checkpoints

## Full Pipeline

The complete reproduction pipeline is:

```bash
conda env create -f environment.yml
conda activate wd

python -m training.prepare

sbatch slurm/train_wordle.sh flat lm
sbatch slurm/train_wordle.sh hrl lm
sbatch slurm/train_wordle.sh flat random
sbatch slurm/train_wordle.sh hrl random

# Wait for training jobs to finish
python -m training.plot
python -m probing.generate_fixed_states

sbatch slurm/extract.sh

# Wait for extraction jobs to finish
sbatch slurm/fit.sh

# Wait for probe fitting to finish
python -m probing.plot
```

