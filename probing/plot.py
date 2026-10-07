import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


CONCEPTS = [
    ("log_num_candidates", "Candidate uncertainty"),
    ("timestep", "Timestep"),
    ("green_count", "Green count"),
]


def plot_flat(ax, d, ylabel=None):
    for prior in ["lm", "random"]:
        x = d[
            (d.architecture == "flat") &
            (d.prior == prior)
        ].sort_values("iteration")

        ax.plot(
            x.iteration,
            x.score,
            marker="o",
            markersize=3,
            label=prior.upper(),
        )

    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(alpha=0.2)


def plot_hrl(ax, d):
    for prior in ["lm", "random"]:
        for rep in ["high", "strategy", "low"]:
            x = d[
                (d.architecture == "hrl") &
                (d.prior == prior) &
                (d.representation == rep)
            ].sort_values("iteration")

            ax.plot(
                x.iteration,
                x.score,
                marker="o",
                markersize=3,
                label=f"{prior.upper()} {rep}",
            )

    ax.grid(alpha=0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="artifacts/probe_results.csv",
    )
    parser.add_argument(
        "--output",
        default="artifacts/probe_plots/probes_all.png",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    fig, axes = plt.subplots(
        4, 2,
        figsize=(15, 17),
        sharex=True,
    )

    # Continuous concepts
    for row, (concept, title) in enumerate(CONCEPTS):
        d = df[df.concept == concept]

        plot_flat(
            axes[row, 0],
            d,
            ylabel=r"Test $R^2$",
        )
        plot_hrl(axes[row, 1], d)

        axes[row, 0].axhline(
            0,
            linestyle="--",
            linewidth=1,
        )
        axes[row, 1].axhline(
            0,
            linestyle="--",
            linewidth=1,
        )

        axes[row, 0].set_title(
            f"Flat — {title}"
        )
        axes[row, 1].set_title(
            f"HRL — {title}"
        )

    # Green positions
    d = df[
        df.concept.str.startswith("green_position_")
    ].copy()

    d = (
        d.groupby(
            [
                "condition",
                "architecture",
                "prior",
                "iteration",
                "representation",
            ],
            as_index=False,
        )
        .score.mean()
    )

    plot_flat(
        axes[3, 0],
        d,
        ylabel="Mean balanced accuracy",
    )
    plot_hrl(axes[3, 1], d)

    axes[3, 0].axhline(
        0.5,
        linestyle="--",
        linewidth=1,
    )
    axes[3, 1].axhline(
        0.5,
        linestyle="--",
        linewidth=1,
    )

    axes[3, 0].set_title(
        "Flat — Green-position information"
    )
    axes[3, 1].set_title(
        "HRL — Green-position information"
    )

    axes[3, 0].set_xlabel("Training iteration")
    axes[3, 1].set_xlabel("Training iteration")

    # One legend per architecture
    axes[0, 0].legend(
        title="Prior",
        loc="best",
    )

    axes[0, 1].legend(
        title="Prior / representation",
        ncol=2,
        loc="best",
    )

    fig.suptitle(
        "Linear Probing Across Training",
        fontsize=16,
        y=0.995,
    )

    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"Saved -> {output}")
    plt.show()


if __name__ == "__main__":
    main()