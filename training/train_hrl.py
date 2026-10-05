# training/train_hrl.py

import argparse
import json
from pathlib import Path
from training.logging_utils import (
    LearningCurveLogger,
)
from training.callbacks import (
    WordleMetricsCallback,
)
from tqdm import tqdm
import ray
import torch

from ray.rllib.algorithms.ppo import (
    PPOConfig,
)

from ray.rllib.core.rl_module.rl_module import (
    RLModuleSpec,
)

from ray.rllib.core.rl_module.multi_rl_module import (
    MultiRLModuleSpec,
)

from ray.tune.registry import (
    register_env,
)

from models.hrl import (
    WordleHighLevelModule,
    WordleLowLevelModule,
)

from training.wordle_hrl_env import (
    HierarchicalWordleEnv,
    HIGH,
    LOW,
)


def policy_mapping_fn(
    agent_id,
    episode,
    **kwargs,
):
    return agent_id


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--prior",
        choices=[
            "lm",
            "random",
        ],
        required=True,
    )

    parser.add_argument(
        "--num-options",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
    )

    args = parser.parse_args()

    # ========================================================
    # GPU
    # ========================================================

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Run inside a GPU allocation."
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    # ========================================================
    # Data
    # ========================================================

    with open(
        "artifacts/splits.json",
        "r",
        encoding="utf-8",
    ) as f:

        splits = json.load(f)

    vocabulary = splits[
        "vocabulary"
    ]

    train_targets = splits[
        "train_targets"
    ]

    embedding_path = Path(
        f"artifacts/"
        f"candidate_embeddings_"
        f"{args.prior}.pt"
    ).resolve()

    embeddings = torch.load(
        embedding_path,
        map_location="cpu",
        weights_only=True,
    )

    word_dim = embeddings.shape[-1]

    # ========================================================
    # Environment
    # ========================================================

    register_env(
        "HierarchicalWordle-v0",

        lambda cfg:
            HierarchicalWordleEnv(cfg),
    )

    # ========================================================
    # Modules
    # ========================================================

    module_spec = MultiRLModuleSpec(

        rl_module_specs={

            HIGH:
                RLModuleSpec(

                    module_class=
                        WordleHighLevelModule,

                    model_config={
                        "hidden_dim":
                            256,
                    },
                ),

            LOW:
                RLModuleSpec(

                    module_class=
                        WordleLowLevelModule,

                    model_config={
                        "hidden_dim":
                            256,

                        "projection_dim":
                            256,

                        "word_dim":
                            word_dim,

                        "candidate_embeddings_path":
                            str(
                                embedding_path
                            ),
                    },
                ),
        }
    )

    # ========================================================
    # PPO
    # ========================================================

    config = (
        PPOConfig()

        .environment(
            "HierarchicalWordle-v0",

            env_config={
                "vocabulary": vocabulary,
                "targets": train_targets,
                "max_guesses": 6,

                "step_penalty": -0.05,
                "yellow_reward": 0.05,
                "green_reward": 0.15,
                "green_retention_reward": 0.05,
                "info_gain_weight": 0.10,
                "solve_reward": 2.0,

                "debug": args.debug,
            },
        )
        .callbacks(
            WordleMetricsCallback
        )
        
        .multi_agent(
            policies={
                HIGH,
                LOW,
            },

            policy_mapping_fn=
                policy_mapping_fn,

            policies_to_train=[
                HIGH,
                LOW,
            ],

            # Important when comparing sampling budgets:
            # HRL has two agent decisions per Wordle guess.
            count_steps_by=
                "env_steps",
        )

        .rl_module(
            rl_module_spec=
                module_spec
        )

        .training(
            lr=3e-4,

            gamma=0.99,

            lambda_=0.95,

            clip_param=0.2,

            entropy_coeff=0.01,

            vf_loss_coeff=0.5,

            train_batch_size_per_learner=
                4096,

            minibatch_size=
                256,

            num_epochs=
                10,
        )

        .learners(
            num_learners=1,
            num_gpus_per_learner=1,
        )

        .env_runners(
            num_env_runners=1 if args.debug else 4,
            num_envs_per_env_runner=1,
        )

        .debugging(
            seed=args.seed,
        )
    )

    ray.init()

    algo = config.build_algo()

    # ========================================================
    # Output
    # ========================================================

    output_dir = (
        Path("checkpoints")
        / f"hrl_{args.prior}"
        / f"seed_{args.seed}"
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    logger = LearningCurveLogger(
        architecture="hrl",
        prior=args.prior,
        seed=args.seed,
        modules=[
            "high_level",
            "low_level",
        ],
    )
    # ========================================================
    # Train
    # ========================================================


    for iteration in tqdm(range(1, args.iterations + 1)):

        result = algo.train()

        row = logger.log(iteration, result)
        logger.print_summary(row)

        if (
            iteration
            % args.checkpoint_every
            == 0
        ):

            checkpoint_dir = (
                output_dir
                / f"iter_{iteration:04d}"
            )

            algo.save_to_path(
                str(checkpoint_dir)
            )

            print(
                "checkpoint:",
                checkpoint_dir,
            )
    logger.plot()
    
    algo.save_to_path(
        str(
            output_dir / "final"
        )
    )
    

    algo.stop()

    ray.shutdown()


if __name__ == "__main__":
    main()