import argparse
import json
from pathlib import Path

import ray
import torch
from tqdm import tqdm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

from models.rl import FlatWordleRLModule
from models.hrl import HierarchicalWordleRLModule
from training.callbacks import WordleMetricsCallback
from training.logging_utils import LearningCurveLogger
from training.wordle_env import WordleEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["flat", "hrl"], required=True)
    parser.add_argument("--prior", choices=["lm", "random"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=15)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--step-penalty", type=float, default=-0.05)
    parser.add_argument("--yellow-reward", type=float, default=0.03)
    parser.add_argument("--green-reward", type=float, default=0.15)
    parser.add_argument("--info-gain-weight", type=float, default=0.005)
    parser.add_argument("--solve-reward", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--vf-loss-coeff", type=float, default=0.5)
    parser.add_argument("--soft-mask-penalty", type=float, default=8.0)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--max-guesses", type=int, default=6)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run inside a GPU allocation.")
    print("GPU:", torch.cuda.get_device_name(0))

    with open("artifacts/splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)

    vocabulary = splits["vocabulary"]
    train_targets = splits["train_targets"]
    val_targets = splits["val_targets"]
    test_targets = splits["test_targets"]
    all_targets = train_targets + val_targets + test_targets

    embedding_path = Path(f"artifacts/candidate_embeddings_{args.prior}.pt").resolve()
    embeddings = torch.load(embedding_path, map_location="cpu", weights_only=True)
    word_dim = embeddings.shape[-1]

    register_env("Wordle-v0", lambda cfg: WordleEnv(cfg))

    env_config = {
        "vocabulary": vocabulary,
        "targets": train_targets,
        "candidate_targets": all_targets,
        "max_guesses": args.max_guesses,
        "step_penalty": args.step_penalty,
        "yellow_reward": args.yellow_reward,
        "green_reward": args.green_reward,
        "info_gain_weight": args.info_gain_weight,
        "solve_reward": args.solve_reward,
        "soft_mask_penalty": args.soft_mask_penalty,
        "debug": args.debug,
    }

    val_env_config = {
        **env_config,
        "targets": val_targets,
        "soft_mask_penalty": 0.0,
        "debug": False,
    }

    if args.architecture == "flat":
        module_class = FlatWordleRLModule
        model_config = {
            "hidden_dim": 256,
            "projection_dim": 256,
            "word_dim": word_dim,
            "candidate_embeddings_path": str(embedding_path),
        }
    else:
        module_class = HierarchicalWordleRLModule
        model_config = {
            "high_dim": 256,
            "strategy_dim": 128,
            "low_dim": 256,
            "projection_dim": 256,
            "word_dim": word_dim,
            "candidate_embeddings_path": str(embedding_path),
        }

    config = (
        PPOConfig()
        .environment("Wordle-v0", env_config=env_config)
        .callbacks(WordleMetricsCallback)
        .rl_module(rl_module_spec=RLModuleSpec(module_class=module_class, model_config=model_config))
        .training(
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.lambda_,
            clip_param=args.clip_param,
            entropy_coeff=args.entropy_coeff,
            vf_loss_coeff=args.vf_loss_coeff,
            train_batch_size_per_learner=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
        )
        .learners(num_learners=1, num_gpus_per_learner=1)
        .env_runners(
            num_env_runners=1 if args.debug else 4,
            num_envs_per_env_runner=1 if args.debug else 8,
        )
        .evaluation(
            evaluation_interval=args.eval_every,
            evaluation_duration=len(val_targets),
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=1,
            evaluation_config={"env_config": val_env_config, "explore": False},
        )
        .debugging(seed=args.seed)
    )

    ray.init()
    algo = config.build_algo()

    exp_name = (
        f"{args.lr}_{args.entropy_coeff}_{args.num_epochs}_"
        f"{args.yellow_reward}_{args.green_reward}_{args.info_gain_weight}_"
        f"{args.solve_reward}_{args.max_guesses}_{args.soft_mask_penalty}"
    )

    output_dir = (
        Path("checkpoints")
        / f"{args.architecture}_{args.prior}"
        / f"seed_{args.seed}"
        / f"exp_{exp_name}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = LearningCurveLogger(
        architecture=args.architecture,
        prior=args.prior,
        seed=args.seed,
        modules=["default_policy"],
        exp_name=exp_name,
    )

    for iteration in tqdm(range(1, args.iterations + 1)):
        result = algo.train()
        row = logger.log(iteration, result)
        logger.print_summary(row)

        if iteration % args.checkpoint_every == 0:
            checkpoint_dir = output_dir / f"iter_{iteration:04d}"
            algo.save_to_path(str(checkpoint_dir))
            print(f"checkpoint: {checkpoint_dir}", flush=True)

    logger.plot()

    final_dir = output_dir / "final"
    algo.save_to_path(str(final_dir))
    print(f"final checkpoint: {final_dir}", flush=True)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()