import argparse
import json
from pathlib import Path

import numpy as np

from training.wordle_core import MAX_GUESSES, compute_feedback, encode_history


def filter_candidates(candidates, guess, feedback):
    observed = tuple(int(x) for x in feedback)
    return [
        word for word in candidates
        if tuple(int(x) for x in compute_feedback(guess, word)) == observed
    ]


def get_concepts(guesses, feedbacks, candidates, prev_num_candidates=None):
    green_positions = np.zeros(5, dtype=np.float32)
    known_letters = np.zeros(26, dtype=np.float32)
    excluded_letters = np.zeros(26, dtype=np.float32)

    for guess, feedback in zip(guesses, feedbacks):
        for pos, (letter, fb) in enumerate(zip(guess, feedback)):
            fb = int(fb)
            idx = ord(letter) - ord("a")

            if fb == 2:
                green_positions[pos] = 1.0
                known_letters[idx] = 1.0
            elif fb == 1:
                known_letters[idx] = 1.0

    for letter in "abcdefghijklmnopqrstuvwxyz":
        idx = ord(letter) - ord("a")
        if all(letter not in word for word in candidates):
            excluded_letters[idx] = 1.0

    num_candidates = len(candidates)

    if prev_num_candidates is None:
        info_gain = 0.0
    else:
        info_gain = (
            np.log(max(prev_num_candidates, 1))
            - np.log(max(num_candidates, 1))
        )

    return {
        "timestep": len(guesses),
        "num_candidates": num_candidates,
        "log_num_candidates": float(np.log(max(num_candidates, 1))),
        "green_count": int(green_positions.sum()),
        "yellow_count": int(sum(
            int(fb) == 1
            for feedback in feedbacks
            for fb in feedback
        )),
        "green_positions": green_positions,
        "known_letters": known_letters,
        "excluded_letters": excluded_letters,
        "info_gain": float(info_gain),
    }


def generate_states(targets, candidate_targets, seed, max_guesses=MAX_GUESSES):
    rng = np.random.default_rng(seed)
    states = []

    for target_id, target in enumerate(targets):
        guesses = []
        feedbacks = []
        candidates = list(candidate_targets)

        for _ in range(max_guesses):
            if not candidates:
                break

            prev_num_candidates = len(candidates)

            # Independent fixed behavior policy:
            # sample uniformly from the current legal candidate set.
            guess = candidates[rng.integers(len(candidates))]
            feedback = compute_feedback(guess, target)

            guesses.append(guess)
            feedbacks.append(feedback)

            candidates = filter_candidates(candidates, guess, feedback)

            # Do not include terminal solved states in the probing corpus.
            if guess == target:
                break

            obs = np.asarray(
                encode_history(
                    guesses,
                    feedbacks,
                    max_guesses=max_guesses,
                ),
                dtype=np.float32,
            )

            concepts = get_concepts(
                guesses,
                feedbacks,
                candidates,
                prev_num_candidates,
            )

            states.append({
                "target_id": target_id,
                "target": target,
                "guesses": list(guesses),
                "feedbacks": [
                    list(map(int, fb))
                    for fb in feedbacks
                ],
                "obs": obs,
                **concepts,
            })

    return states


def save_states(states, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        obs=np.stack([s["obs"] for s in states]),
        target_id=np.asarray([s["target_id"] for s in states]),
        timestep=np.asarray([s["timestep"] for s in states]),
        num_candidates=np.asarray([s["num_candidates"] for s in states]),
        log_num_candidates=np.asarray([s["log_num_candidates"] for s in states]),
        green_count=np.asarray([s["green_count"] for s in states]),
        yellow_count=np.asarray([s["yellow_count"] for s in states]),
        green_positions=np.stack([s["green_positions"] for s in states]),
        known_letters=np.stack([s["known_letters"] for s in states]),
        excluded_letters=np.stack([s["excluded_letters"] for s in states]),
        info_gain=np.asarray([s["info_gain"] for s in states]),
    )

    metadata = [
        {
            "target_id": s["target_id"],
            "target": s["target"],
            "guesses": s["guesses"],
            "feedbacks": s["feedbacks"],
        }
        for s in states
    ]

    with open(path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", default="artifacts/splits.json")
    parser.add_argument("--output-dir", default="artifacts/probe_states")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-guesses", type=int, default=MAX_GUESSES)
    args = parser.parse_args()

    with open(args.split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_targets = splits["train_targets"]
    val_targets = splits["val_targets"]
    test_targets = splits["test_targets"]

    candidate_targets = train_targets + val_targets + test_targets
    output_dir = Path(args.output_dir)

    for i, (name, targets) in enumerate([
        ("val", val_targets),
        ("test", test_targets),
    ]):
        states = generate_states(
            targets=targets,
            candidate_targets=candidate_targets,
            seed=args.seed + i,
            max_guesses=args.max_guesses,
        )

        path = output_dir / f"{name}.npz"
        save_states(states, path)

        print(
            f"{name}: {len(targets)} targets, "
            f"{len(states)} non-terminal states -> {path}"
        )


if __name__ == "__main__":
    main()