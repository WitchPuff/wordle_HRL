import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--start", type=int, default=5)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--step", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.checkpoint_root).resolve()

    conditions = [
        ("flat_lm", "flat"),
        ("flat_random", "flat"),
        ("hrl_lm", "hrl"),
        ("hrl_random", "hrl"),
    ]

    total = 0
    failed = []

    for condition, architecture in conditions:
        exp_dir = root / condition / f"seed_{args.seed}" / args.exp_name

        if not exp_dir.exists():
            print(f"[SKIP] Missing experiment: {exp_dir}")
            continue

        for iteration in range(args.start, args.end + 1, args.step):
            checkpoint = exp_dir / f"iter_{iteration:04d}"
            name = f"{condition}/iter_{iteration:04d}"

            if not checkpoint.exists():
                print(f"[SKIP] Missing checkpoint: {checkpoint}")
                continue

            print(f"\n{'=' * 70}")
            print(f"{condition} | iter {iteration}")
            print(f"{'=' * 70}")

            cmd = [
                "python", "-m", "probing.extract_representations",
                "--checkpoint", str(checkpoint),
                "--architecture", architecture,
                "--name", name,
            ]

            result = subprocess.run(cmd)

            if result.returncode != 0:
                failed.append((condition, iteration))
                print(f"[FAILED] {condition} iter {iteration}")
            else:
                total += 1

    print(f"\nFinished: {total} checkpoints extracted.")

    if failed:
        print("Failed:")
        for condition, iteration in failed:
            print(f"  {condition} iter {iteration}")


if __name__ == "__main__":
    main()