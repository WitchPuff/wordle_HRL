# training/wordle_core.py

import numpy as np


WORD_LENGTH = 5
MAX_GUESSES = 6

GRAY = 0
YELLOW = 1
GREEN = 2

LETTER_DIM = 27
FEEDBACK_DIM = 4

TURN_DIM = MAX_GUESSES + 1

OBS_DIM = (
    MAX_GUESSES
    * WORD_LENGTH
    * (LETTER_DIM + FEEDBACK_DIM)
    + TURN_DIM
)


def compute_feedback(
    guess: str,
    target: str,
):
    """
    Standard Wordle feedback with correct
    duplicate-letter handling.
    """

    feedback = np.zeros(
        WORD_LENGTH,
        dtype=np.int64,
    )

    remaining = {}

    # Greens first.
    for i, (g, t) in enumerate(
        zip(guess, target)
    ):
        if g == t:
            feedback[i] = GREEN
        else:
            remaining[t] = (
                remaining.get(t, 0) + 1
            )

    # Yellows second.
    for i, g in enumerate(guess):

        if feedback[i] == GREEN:
            continue

        if remaining.get(g, 0) > 0:
            feedback[i] = YELLOW
            remaining[g] -= 1

    return feedback


def letter_one_hot(char=None):
    """
    27 categories:
        0 = padding
        1..26 = a..z
    """

    x = np.zeros(
        LETTER_DIM,
        dtype=np.float32,
    )

    if char is None:
        x[0] = 1.0
    else:
        idx = ord(char) - ord("a") + 1
        x[idx] = 1.0

    return x


def feedback_one_hot(value=None):
    """
    4 categories:
        0 = padding
        1 = gray
        2 = yellow
        3 = green
    """

    x = np.zeros(
        FEEDBACK_DIM,
        dtype=np.float32,
    )

    if value is None:
        x[0] = 1.0
    else:
        x[int(value) + 1] = 1.0

    return x


def encode_history(
    guesses,
    feedbacks,
    max_guesses=MAX_GUESSES,
):
    """
    Raw categorical Wordle history.

    Does NOT explicitly provide:
        - candidate count
        - known-letter count
        - known-position count
        - entropy
        - solution identity

    These remain potential probing targets.
    """

    encoded = []

    for turn in range(max_guesses):

        if turn < len(guesses):

            guess = guesses[turn]
            feedback = feedbacks[turn]

            for pos in range(WORD_LENGTH):

                encoded.append(
                    letter_one_hot(
                        guess[pos]
                    )
                )

                encoded.append(
                    feedback_one_hot(
                        feedback[pos]
                    )
                )

        else:

            for _ in range(WORD_LENGTH):

                encoded.append(
                    letter_one_hot(None)
                )

                encoded.append(
                    feedback_one_hot(None)
                )

    # Explicit categorical turn marker.
    turn_vector = np.zeros(
        max_guesses + 1,
        dtype=np.float32,
    )

    turn_vector[
        min(len(guesses), max_guesses)
    ] = 1.0

    encoded.append(turn_vector)

    obs = np.concatenate(encoded).astype(
        np.float32
    )

    expected_dim = (
        max_guesses
        * WORD_LENGTH
        * (LETTER_DIM + FEEDBACK_DIM)
        + max_guesses + 1
    )

    assert obs.shape == (expected_dim,)

    return obs