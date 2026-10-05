# training/prepare.py

import json
import random
from pathlib import Path

import torch

from models.lm import (
    CharacterLM,
    precompute_vocabulary_embeddings,
)


SEED = 42

DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_words(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        words = [
            line.strip().lower()
            for line in f
            if line.strip()
        ]

    words = [
        w
        for w in words
        if (
            len(w) == 5
            and w.isascii()
            and w.isalpha()
        )
    ]

    return sorted(set(words))


def main():

    # ========================================================
    # Vocabulary
    # ========================================================

    allowed_words = load_words(
        DATA_DIR / "allowed_words.txt"
    )

    answers = load_words(
        DATA_DIR / "answers.txt"
    )

    # Every answer must also be a legal action.
    allowed_words = sorted(
        set(allowed_words)
        | set(answers)
    )

    print(
        "Allowed actions:",
        len(allowed_words),
    )

    print(
        "Possible answers:",
        len(answers),
    )

    # ========================================================
    # Fixed target split
    # ========================================================

    rng = random.Random(SEED)

    shuffled = answers.copy()

    rng.shuffle(shuffled)

    n = len(shuffled)

    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_targets = shuffled[
        :n_train
    ]

    val_targets = shuffled[
        n_train:
        n_train + n_val
    ]

    test_targets = shuffled[
        n_train + n_val:
    ]

    splits = {
        "split_seed": SEED,
        "vocabulary": allowed_words,
        "train_targets": train_targets,
        "val_targets": val_targets,
        "test_targets": test_targets,
    }

    with open(
        ARTIFACT_DIR / "splits.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            splits,
            f,
            indent=2,
        )

    # ========================================================
    # LM embeddings
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Embedding device:", device)

    lm = CharacterLM(
        model_name="google/canine-s",
        freeze=True,
    ).to(device)

    lm_embeddings = (
        precompute_vocabulary_embeddings(
            lm,
            allowed_words,
            batch_size=256,
        )
        .float()
        .cpu()
    )

    torch.save(
        lm_embeddings,
        ARTIFACT_DIR
        / "candidate_embeddings_lm.pt",
    )

    # ========================================================
    # Random prior control
    # ========================================================

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(SEED)

    random_embeddings = torch.randn(
        lm_embeddings.shape,
        generator=generator,
    )

    # Match global scale approximately.
    lm_std = lm_embeddings.std()

    random_embeddings = (
        random_embeddings
        / random_embeddings.std()
        * lm_std
    )

    torch.save(
        random_embeddings,
        ARTIFACT_DIR
        / "candidate_embeddings_random.pt",
    )

    print(
        "Embedding shape:",
        tuple(lm_embeddings.shape),
    )

    print("Preparation complete.")


if __name__ == "__main__":
    main()