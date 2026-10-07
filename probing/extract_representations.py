import argparse
from pathlib import Path

import numpy as np
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from training.wordle_env import WordleEnv

def load_states(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def get_module(algo):
    # RLlib new API stack
    try:
        return algo.get_module()
    except (AttributeError, TypeError):
        return algo.get_module("default_policy")


def extract_flat(module, obs, batch_size, device):
    reps = []

    module.eval()

    with torch.no_grad():
        for start in range(0, len(obs), batch_size):
            x = torch.as_tensor(
                obs[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )

            h = module.get_representation(x)
            reps.append(h.detach().cpu().numpy())

    return {"h": np.concatenate(reps, axis=0)}


def extract_hrl(module, obs, batch_size, device):
    high_reps = []
    strategies = []
    low_reps = []

    module.eval()

    with torch.no_grad():
        for start in range(0, len(obs), batch_size):
            x = torch.as_tensor(
                obs[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )

            high, strategy = module.encode_high(x)
            low = module.encode_low(x, strategy)

            high_reps.append(high.detach().cpu().numpy())
            strategies.append(strategy.detach().cpu().numpy())
            low_reps.append(low.detach().cpu().numpy())

    return {
        "high": np.concatenate(high_reps, axis=0),
        "strategy": np.concatenate(strategies, axis=0),
        "low": np.concatenate(low_reps, axis=0),
    }


def save_representations(reps, states, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = {
        k: states[k]
        for k in [
            "target_id",
            "timestep",
            "num_candidates",
            "log_num_candidates",
            "green_count",
            "yellow_count",
            "green_positions",
            "known_letters",
            "excluded_letters",
            "info_gain",
        ]
        if k in states
    }

    np.savez_compressed(
        output_path,
        **reps,
        **labels,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--architecture",
        choices=["flat", "hrl"],
        required=True,
    )
    parser.add_argument(
        "--state-dir",
        default="artifacts/probe_states",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/probe_representations",
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch-size", type=int, default=256)

    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"Loading checkpoint: {checkpoint}")

    register_env("Wordle", lambda config: WordleEnv(config))
    register_env("Wordle-v0", lambda config: WordleEnv(config))
    algo = Algorithm.from_checkpoint(str(checkpoint))
    module = get_module(algo)

    device = next(module.parameters()).device

    print(f"Architecture: {args.architecture}")
    print(f"Device: {device}")

    for split in ["val", "test"]:
        state_path = Path(args.state_dir) / f"{split}.npz"
        states = load_states(state_path)
        obs = states["obs"]

        print(f"{split}: {len(obs)} states")

        if args.architecture == "flat":
            reps = extract_flat(
                module,
                obs,
                args.batch_size,
                device,
            )
        else:
            reps = extract_hrl(
                module,
                obs,
                args.batch_size,
                device,
            )

        output_path = (
            Path(args.output_dir)
            / args.name
            / f"{split}.npz"
        )

        save_representations(
            reps,
            states,
            output_path,
        )

        print(f"Saved -> {output_path}")

        for key, value in reps.items():
            print(f"  {key}: {value.shape}")

    algo.stop()


if __name__ == "__main__":
    main()