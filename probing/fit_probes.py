import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, balanced_accuracy_score


CONTINUOUS = ["log_num_candidates", "timestep", "green_count"]

ALPHAS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
CS = [0.01, 0.1, 1.0, 10.0, 100.0]


def fit_ridge(X_train, y_train, X_test, y_test, groups):
    model = GridSearchCV(
        make_pipeline(StandardScaler(), Ridge()),
        {"ridge__alpha": ALPHAS},
        cv=GroupKFold(n_splits=5),
        scoring="r2",
        n_jobs=-1,
    )

    model.fit(X_train, y_train, groups=groups)
    pred = model.predict(X_test)

    return (
        r2_score(y_test, pred),
        model.best_params_["ridge__alpha"],
    )


def fit_logistic(X_train, y_train, X_test, y_test, groups):
    model = GridSearchCV(
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
            ),
        ),
        {"logisticregression__C": CS},
        cv=StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        ),
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    model.fit(X_train, y_train, groups=groups)
    pred = model.predict(X_test)

    return (
        balanced_accuracy_score(y_test, pred),
        model.best_params_["logisticregression__C"],
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="artifacts/probe_representations",
    )

    parser.add_argument(
        "--output",
        default="artifacts/probe_results.csv",
    )

    args = parser.parse_args()

    root = Path(args.root)

    conditions = [
        "flat_lm",
        "flat_random",
        "hrl_lm",
        "hrl_random",
    ]

    rows = []

    for condition in conditions:

        architecture = (
            "hrl"
            if condition.startswith("hrl")
            else "flat"
        )

        prior = (
            "lm"
            if condition.endswith("_lm")
            else "random"
        )

        representations = (
            ["high", "strategy", "low"]
            if architecture == "hrl"
            else ["h"]
        )

        for iteration in range(5, 101, 5):

            checkpoint = (
                root
                / condition
                / f"iter_{iteration:04d}"
            )

            val_path = checkpoint / "val.npz"
            test_path = checkpoint / "test.npz"

            if (
                not val_path.exists()
                or not test_path.exists()
            ):
                print(
                    f"[SKIP] "
                    f"{condition} iter {iteration}"
                )
                continue

            val = np.load(val_path)
            test = np.load(test_path)

            # Important:
            # all states originating from the same
            # Wordle target stay in the same CV fold.
            groups = val["target_id"]

            for representation in representations:

                X_train = val[representation]
                X_test = test[representation]

                print(
                    f"{condition:12s} "
                    f"iter={iteration:3d} "
                    f"rep={representation:8s}"
                )

                # -------------------------
                # Continuous concepts
                # -------------------------

                for concept in CONTINUOUS:

                    score, alpha = fit_ridge(
                        X_train,
                        val[concept],
                        X_test,
                        test[concept],
                        groups,
                    )

                    rows.append({
                        "condition": condition,
                        "architecture": architecture,
                        "prior": prior,
                        "iteration": iteration,
                        "representation": representation,
                        "concept": concept,
                        "probe": "ridge",
                        "metric": "r2",
                        "score": score,
                        "hyperparameter": alpha,
                    })

                # -------------------------
                # Green position concepts
                # -------------------------

                for pos in range(5):

                    y_train = (
                        val["green_positions"][:, pos]
                        .astype(int)
                    )

                    y_test = (
                        test["green_positions"][:, pos]
                        .astype(int)
                    )

                    if len(np.unique(y_train)) < 2:
                        print(
                            f"  [SKIP] "
                            f"green_position_{pos}: "
                            f"one class"
                        )
                        continue

                    score, C = fit_logistic(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        groups,
                    )

                    rows.append({
                        "condition": condition,
                        "architecture": architecture,
                        "prior": prior,
                        "iteration": iteration,
                        "representation": representation,
                        "concept": f"green_position_{pos}",
                        "probe": "logistic",
                        "metric": "balanced_accuracy",
                        "score": score,
                        "hyperparameter": C,
                    })

    df = pd.DataFrame(rows)

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output,
        index=False,
    )

    print(
        f"\nSaved {len(df)} probe results "
        f"-> {output}"
    )


if __name__ == "__main__":
    main()