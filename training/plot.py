import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PARAMS = "0.0001_0.01_5_0.0_0.0_0.0_10.0_6_8.0"
ROOT = Path("results")

EXPERIMENTS = {
    "Flat + LM": "flat_lm",
    "Flat + Random": "flat_random",
    "HRL + LM": "hrl_lm",
    "HRL + Random": "hrl_random",
}

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
axes = axes.flatten()

for ax, (title, condition) in zip(axes, EXPERIMENTS.items()):
    df = pd.read_csv(ROOT / condition / PARAMS / "seed_42.csv")
    x = df["iteration"]

    l1, = ax.plot(x, df["solve_rate"], marker=".", label="Solve rate")
    ax.set_title(title)
    ax.set_ylabel("Solve rate")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)

    ax2 = ax.twinx()
    l2, = ax2.plot(
        x, df["mean_guesses_solved"],
        marker=".", linestyle="--", label="Mean guesses", color='purple'
    )
    ax2.set_ylabel("Mean guesses (solved)")
    ax2.set_ylim(1, 6)

    ax.legend(
        [l1, l2],
        ["Solve rate", "Mean guesses"],
        loc="best"
    )

for ax in axes[-2:]:
    ax.set_xlabel("Training iteration")

fig.suptitle("Training Performance", fontsize=16)
fig.tight_layout()
fig.savefig("training_performance.png", dpi=200, bbox_inches="tight")
plt.show()