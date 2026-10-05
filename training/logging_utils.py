# training/logging_utils.py

import csv
import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
# ============================================================
# Helpers
# ============================================================

def _safe_get(
    dictionary,
    key,
    default=math.nan,
):
    """
    Safely retrieve a value from a dictionary.
    """

    if not isinstance(
        dictionary,
        dict,
    ):
        return default

    value = dictionary.get(
        key,
        default,
    )

    if value is None:
        return default

    return value


def _get_env_metric(
    env_metrics,
    key,
    default=math.nan,
):
    """
    Retrieve a custom environment metric.

    First checks:

        env_runners[key]

    Then:

        env_runners["custom_metrics"][key]

    This makes the logger slightly more robust to
    differences in RLlib metric nesting.
    """

    if not isinstance(
        env_metrics,
        dict,
    ):
        return default

    # --------------------------------------------------------
    # Direct metric
    # --------------------------------------------------------

    if key in env_metrics:

        value = env_metrics[
            key
        ]

        if value is not None:
            return value

    # --------------------------------------------------------
    # custom_metrics
    # --------------------------------------------------------

    custom_metrics = env_metrics.get(
        "custom_metrics",
        {},
    )

    if (
        isinstance(
            custom_metrics,
            dict,
        )
        and key in custom_metrics
    ):

        value = custom_metrics[
            key
        ]

        if value is not None:
            return value

    return default


# ============================================================
# Learning Curve Logger
# ============================================================

class LearningCurveLogger:
    """
    Shared CSV logger for Flat RL and HRL.

    Flat example:

        logger = LearningCurveLogger(
            architecture="flat",
            prior="lm",
            seed=42,
            modules=["default_policy"],
        )

    HRL example:

        logger = LearningCurveLogger(
            architecture="hrl",
            prior="lm",
            seed=42,
            modules=[
                "high_level",
                "low_level",
            ],
        )

    One row is written after every PPO iteration.
    """

    # ========================================================
    # Environment / task-level metrics
    # ========================================================
    ENV_FIELDS = [
        "iteration",
        "episode_return",
        "episode_length",
        "solve_rate",
        "mean_guesses_solved",

        "mean_info_gain",
        "mean_new_yellows",
        "mean_new_greens",
        "mean_retained_greens",
        "mean_lost_greens",
        "mean_final_candidates"
    ]

    # ========================================================
    # PPO learner metrics
    # ========================================================

    LEARNER_FIELDS = [

        "total_loss",

        "policy_loss",

        "value_loss",

        "entropy",

        "kl",

        "explained_variance",

        "learning_rate",
    ]

    # ========================================================
    # Initialization
    # ========================================================

    def __init__(
        self,
        architecture,
        prior,
        seed,
        modules,
        results_root="results",
        exp_name=None
    ):

        self.architecture = str(
            architecture
        )

        self.prior = str(
            prior
        )

        self.seed = int(
            seed
        )

        self.modules = list(
            modules
        )

        # ----------------------------------------------------
        # Output directory
        #
        # results/
        #
        #   flat_lm/
        #       seed_42.csv
        #
        #   flat_random/
        #       seed_42.csv
        #
        #   hrl_lm/
        #       seed_42.csv
        #
        #   hrl_random/
        #       seed_42.csv
        # ----------------------------------------------------

        self.results_dir = (
            Path(
                results_root
            )
            / (
                f"{self.architecture}"
                f"_{self.prior}"
            )
            / (exp_name if exp_name is not None else "")
        )

        self.results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.csv_path = (
            self.results_dir
            / f"seed_{self.seed}.csv"
        )

        # ----------------------------------------------------
        # Build CSV columns
        # ----------------------------------------------------

        self.fieldnames = list(
            self.ENV_FIELDS
        )

        for module in self.modules:

            for metric in (
                self.LEARNER_FIELDS
            ):

                self.fieldnames.append(
                    f"{module}_{metric}"
                )

        # ----------------------------------------------------
        # Create / overwrite CSV
        # ----------------------------------------------------

        with open(
            self.csv_path,
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=self.fieldnames,
            )

            writer.writeheader()

        print(
            "\n"
            "Learning curve logger initialized:"
        )

        print(
            self.csv_path.resolve(),
            flush=True,
        )

    # ========================================================
    # Learner metric extraction
    # ========================================================

    def _extract_learner_metrics(
        self,
        learner,
    ):

        return {

            "total_loss":
                _safe_get(
                    learner,
                    "total_loss",
                ),

            "policy_loss":
                _safe_get(
                    learner,
                    "policy_loss",
                ),

            "value_loss":
                _safe_get(
                    learner,
                    "vf_loss",
                ),

            "entropy":
                _safe_get(
                    learner,
                    "entropy",
                ),

            "kl":
                _safe_get(
                    learner,
                    "mean_kl_loss",
                ),

            "explained_variance":
                _safe_get(
                    learner,
                    "vf_explained_var",
                ),

            "learning_rate":
                _safe_get(
                    learner,
                    "default_optimizer_learning_rate",
                ),
        }

    # ========================================================
    # Log one PPO iteration
    # ========================================================

    def log(
        self,
        iteration,
        result,
    ):

        env_metrics = result.get(
            "env_runners",
            {},
        )

        learners = result.get(
            "learners",
            {},
        )

        # ----------------------------------------------------
        # Callback metrics
        # ----------------------------------------------------

        solve_rate = _get_env_metric(
            env_metrics,
            "solve_rate",
        )

        mean_guesses_solved = _get_env_metric(
            env_metrics,
            "mean_guesses_solved",
        )

        mean_info_gain = _get_env_metric(
            env_metrics,
            "mean_info_gain",
        )

        mean_new_yellows = _get_env_metric(
            env_metrics,
            "mean_new_yellows",
        )

        mean_new_greens = _get_env_metric(
            env_metrics,
            "mean_new_greens",
        )

        mean_retained_greens = _get_env_metric(
            env_metrics,
            "mean_retained_greens",
        )
        mean_lost_greens = _get_env_metric(
            env_metrics,
            "mean_lost_greens",
        )
        
        mean_final_candidates = _get_env_metric(
            env_metrics,
            "mean_final_candidates",
        )

        # ----------------------------------------------------
        # Task-level metrics
        # ----------------------------------------------------

        row = {

            "iteration":
                int(iteration),

            "episode_return":
                _safe_get(
                    env_metrics,
                    "episode_return_mean",
                ),

            "episode_length":
                _safe_get(
                    env_metrics,
                    "episode_len_mean",
                ),

            "solve_rate":
                solve_rate,

            "mean_guesses_solved":
                mean_guesses_solved,

            "mean_info_gain":
                mean_info_gain,

            "mean_new_yellows":
                mean_new_yellows,

            "mean_new_greens":
                mean_new_greens,

            "mean_retained_greens":
                mean_retained_greens,
            
            "mean_lost_greens": mean_lost_greens,
                
            "mean_final_candidates":
                mean_final_candidates,
        }

        # ----------------------------------------------------
        # Learner metrics
        # ----------------------------------------------------

        for module in self.modules:

            learner = learners.get(
                module,
                {},
            )

            metrics = self._extract_learner_metrics(
                learner
            )

            for metric_name, value in metrics.items():

                row[
                    f"{module}_{metric_name}"
                ] = value

        with open(
            self.csv_path,
            "a",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=self.fieldnames,
            )

            writer.writerow(row)

        return row
        # ========================================================
        # Pretty-print current iteration
        # ========================================================

    def print_summary(
        self,
        row,
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"{self.architecture.upper()} "
            f"| "
            f"{self.prior.upper()} "
            f"| "
            f"seed={self.seed} "
            f"| "
            f"iteration="
            f"{row['iteration']}"
        )

        print(
            "-" * 70
        )

        print(
            "Task"
        )

        print(
            f"  episode return      : "
            f"{row['episode_return']}"
        )

        print(
            f"  episode length      : "
            f"{row['episode_length']}"
        )

        print(
            f"  solve rate          : "
            f"{row['solve_rate']}"
        )

        print(
            f"  mean guesses solved : "
            f"{row['mean_guesses_solved']}"
        )

        print(
            f"  mean info gain      : "
            f"{row['mean_info_gain']}"
        )

        print(
            f"  mean new yellows    : "
            f"{row['mean_new_yellows']}"
        )

        print(
            f"  mean new greens     : "
            f"{row['mean_new_greens']}"
        )

        print(
            f"  mean retained greens: "
            f"{row['mean_retained_greens']}"
        )
        
        print(
            f"  mean final candidates: "
            f"{row['mean_final_candidates']}"
        )

        # ----------------------------------------------------
        # Module-specific metrics
        # ----------------------------------------------------

        for module in self.modules:

            print(
                "\n"
                f"{module}"
            )

            for metric in (
                self.LEARNER_FIELDS
            ):

                key = (
                    f"{module}"
                    f"_{metric}"
                )

                print(
                    f"  "
                    f"{metric:<18}: "
                    f"{row.get(key)}"
                )

        print(
            "=" * 70,
            flush=True,
        )
    # ========================================================
    # Plot learning curves
    # ========================================================

    def plot(
        self,
        output_dir=None,
        show=False,
    ):
        """
        Plot learning curves from the CSV file.

        Generates:
            1. iteration vs solve rate
            2. iteration vs mean guesses (solved episodes)
            3. iteration vs mean episode return

        Figures are saved as PNG files.
        """

        # ----------------------------------------------------
        # Load CSV
        # ----------------------------------------------------

        df = pd.read_csv(
            self.csv_path
        )

        if len(df) == 0:
            print(
                "No data available to plot.",
                flush=True,
            )
            return

        # ----------------------------------------------------
        # Output directory
        # ----------------------------------------------------

        if output_dir is None:

            output_dir = (
                self.results_dir
                / "plots"
            )

        else:

            output_dir = Path(
                output_dir
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        title_prefix = (
            f"{self.architecture.upper()} "
            f"+ {self.prior.upper()} "
            f"(seed {self.seed})"
        )

        # ====================================================
        # 1. Solve rate
        # ====================================================

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.plot(
            df["iteration"],
            df["solve_rate"],
            marker="o",
            markersize=3,
        )

        ax.set_xlabel(
            "PPO iteration"
        )

        ax.set_ylabel(
            "Solve rate"
        )

        ax.set_ylim(
            0.0,
            1.0,
        )

        ax.set_title(
            f"{title_prefix}\n"
            "Solve rate"
        )

        ax.grid(
            alpha=0.3
        )

        fig.tight_layout()

        solve_path = (
            output_dir
            / "solve_rate.png"
        )

        fig.savefig(
            solve_path,
            dpi=200,
        )

        if show:
            plt.show()

        plt.close(
            fig
        )

        # ====================================================
        # 2. Mean guesses among solved episodes
        # ====================================================

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.plot(
            df["iteration"],
            df["mean_guesses_solved"],
            marker="o",
            markersize=3,
        )

        ax.set_xlabel(
            "PPO iteration"
        )

        ax.set_ylabel(
            "Mean guesses (solved)"
        )

        ax.set_ylim(
            1.0,
            6.0,
        )

        ax.set_title(
            f"{title_prefix}\n"
            "Mean guesses among solved episodes"
        )

        ax.grid(
            alpha=0.3
        )

        fig.tight_layout()

        guesses_path = (
            output_dir
            / "mean_guesses_solved.png"
        )

        fig.savefig(
            guesses_path,
            dpi=200,
        )

        if show:
            plt.show()

        plt.close(
            fig
        )

        # ====================================================
        # 3. Episode return
        # ====================================================

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.plot(
            df["iteration"],
            df["episode_return"],
            marker="o",
            markersize=3,
        )

        ax.set_xlabel(
            "PPO iteration"
        )

        ax.set_ylabel(
            "Mean episode return"
        )

        ax.set_title(
            f"{title_prefix}\n"
            "Episode return"
        )

        ax.grid(
            alpha=0.3
        )

        fig.tight_layout()

        reward_path = (
            output_dir
            / "episode_return.png"
        )

        fig.savefig(
            reward_path,
            dpi=200,
        )

        if show:
            plt.show()

        plt.close(
            fig
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print(
            "\nLearning curves saved:",
            flush=True,
        )

        print(
            f"  {solve_path.resolve()}",
            flush=True,
        )

        print(
            f"  {guesses_path.resolve()}",
            flush=True,
        )

        print(
            f"  {reward_path.resolve()}",
            flush=True,
        )