# training/logging_utils.py

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _safe_get(d, key, default=math.nan):
    if not isinstance(d, dict):
        return default
    value = d.get(key, default)
    return default if value is None else value


def _get_env_metric(env_metrics, key, default=math.nan):
    if not isinstance(env_metrics, dict):
        return default

    if key in env_metrics and env_metrics[key] is not None:
        return env_metrics[key]

    custom = env_metrics.get("custom_metrics", {})
    if isinstance(custom, dict) and key in custom and custom[key] is not None:
        return custom[key]

    return default


class LearningCurveLogger:
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
        "mean_final_candidates",
    ]

    VAL_FIELDS = [
        "val_solve_rate",
        "val_episode_return",
        "val_episode_length",
        "val_mean_guesses_solved",
    ]

    LEARNER_FIELDS = [
        "total_loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "kl",
        "explained_variance",
        "learning_rate",
    ]

    def __init__(self, architecture, prior, seed, modules, results_root="results", exp_name=None):
        self.architecture = str(architecture)
        self.prior = str(prior)
        self.seed = int(seed)
        self.modules = list(modules)

        self.results_dir = Path(results_root) / f"{self.architecture}_{self.prior}"
        if exp_name is not None:
            self.results_dir /= exp_name

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.results_dir / f"seed_{self.seed}.csv"

        self.fieldnames = self.ENV_FIELDS + self.VAL_FIELDS
        for module in self.modules:
            self.fieldnames += [f"{module}_{metric}" for metric in self.LEARNER_FIELDS]

        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

        print(f"\nLearning curve logger initialized:\n{self.csv_path.resolve()}", flush=True)

    def _extract_learner_metrics(self, learner):
        return {
            "total_loss": _safe_get(learner, "total_loss"),
            "policy_loss": _safe_get(learner, "policy_loss"),
            "value_loss": _safe_get(learner, "vf_loss"),
            "entropy": _safe_get(learner, "entropy"),
            "kl": _safe_get(learner, "mean_kl_loss"),
            "explained_variance": _safe_get(learner, "vf_explained_var"),
            "learning_rate": _safe_get(learner, "default_optimizer_learning_rate"),
        }

    def log(self, iteration, result):
        env = result.get("env_runners", {})
        val = result.get("evaluation", {}).get("env_runners", {})
        learners = result.get("learners", {})

        row = {
            "iteration": int(iteration),
            "episode_return": _safe_get(env, "episode_return_mean"),
            "episode_length": _safe_get(env, "episode_len_mean"),
            "solve_rate": _get_env_metric(env, "solve_rate"),
            "mean_guesses_solved": _get_env_metric(env, "mean_guesses_solved"),
            "mean_info_gain": _get_env_metric(env, "mean_info_gain"),
            "mean_new_yellows": _get_env_metric(env, "mean_new_yellows"),
            "mean_new_greens": _get_env_metric(env, "mean_new_greens"),
            "mean_retained_greens": _get_env_metric(env, "mean_retained_greens"),
            "mean_lost_greens": _get_env_metric(env, "mean_lost_greens"),
            "mean_final_candidates": _get_env_metric(env, "mean_final_candidates"),
            "val_solve_rate": _get_env_metric(val, "solve_rate"),
            "val_episode_return": _safe_get(val, "episode_return_mean"),
            "val_episode_length": _safe_get(val, "episode_len_mean"),
            "val_mean_guesses_solved": _get_env_metric(val, "mean_guesses_solved"),
        }

        for module in self.modules:
            metrics = self._extract_learner_metrics(learners.get(module, {}))
            for name, value in metrics.items():
                row[f"{module}_{name}"] = value

        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)

        return row
    def print_summary(self, row):
        print("\n" + "=" * 70)
        print(f"{self.architecture.upper()} | {self.prior.upper()} | seed={self.seed} | iteration={row['iteration']}")
        print("-" * 70)

        print(
            f"TRAIN | return={row['episode_return']:.3f} | "
            f"length={row['episode_length']:.2f} | "
            f"solve={row['solve_rate']:.3f} | "
            f"guesses={row['mean_guesses_solved']:.2f}"
        )

        print(
            f"      | info={row['mean_info_gain']:.3f} | "
            f"yellow={row['mean_new_yellows']:.3f} | "
            f"green={row['mean_new_greens']:.3f} | "
            f"retained={row['mean_retained_greens']:.3f} | "
            f"lost={row['mean_lost_greens']:.3f} | "
            f"candidates={row['mean_final_candidates']:.3f}"
        )

        if not math.isnan(row["val_solve_rate"]):
            print(
                f"VAL   | return={row['val_episode_return']:.3f} | "
                f"length={row['val_episode_length']:.2f} | "
                f"solve={row['val_solve_rate']:.3f} | "
                f"guesses={row['val_mean_guesses_solved']:.2f}"
            )

        for module in self.modules:
            print(f"\n{module}")
            for metric in self.LEARNER_FIELDS:
                print(f"  {metric:<18}: {row.get(f'{module}_{metric}')}")

        print("=" * 70, flush=True)

    def plot(self, output_dir=None, show=False):
        df = pd.read_csv(self.csv_path)

        if df.empty:
            print("No data available to plot.", flush=True)
            return

        output_dir = Path(output_dir) if output_dir else self.results_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        title = f"{self.architecture.upper()} + {self.prior.upper()} (seed {self.seed})"

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(df["iteration"], df["solve_rate"], marker="o", markersize=3, label="Train")

        val = df.dropna(subset=["val_solve_rate"])
        if not val.empty:
            ax.plot(val["iteration"], val["val_solve_rate"], marker="o", markersize=4, label="Validation")

        ax.set(xlabel="PPO iteration", ylabel="Solve rate", ylim=(0, 1), title=f"{title}\nSolve rate")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        solve_path = output_dir / "solve_rate.png"
        fig.savefig(solve_path, dpi=200)

        if show:
            plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(df["iteration"], df["mean_guesses_solved"], marker="o", markersize=3, label="Train")

        if not val.empty:
            ax.plot(val["iteration"], val["val_mean_guesses_solved"], marker="o", markersize=4, label="Validation")

        ax.set(xlabel="PPO iteration", ylabel="Mean guesses (solved)", ylim=(1, 6), title=f"{title}\nMean guesses")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        guesses_path = output_dir / "mean_guesses_solved.png"
        fig.savefig(guesses_path, dpi=200)

        if show:
            plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(df["iteration"], df["episode_return"], marker="o", markersize=3, label="Train")

        if not val.empty:
            ax.plot(val["iteration"], val["val_episode_return"], marker="o", markersize=4, label="Validation")

        ax.set(xlabel="PPO iteration", ylabel="Mean episode return", title=f"{title}\nEpisode return")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        reward_path = output_dir / "episode_return.png"
        fig.savefig(reward_path, dpi=200)

        if show:
            plt.show()
        plt.close(fig)

        print(
            f"\nLearning curves saved:\n"
            f"  {solve_path.resolve()}\n"
            f"  {guesses_path.resolve()}\n"
            f"  {reward_path.resolve()}",
            flush=True,
        )