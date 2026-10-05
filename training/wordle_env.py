import numpy as np
import gymnasium as gym
from gymnasium import spaces

from training.wordle_core import (
    MAX_GUESSES,
    OBS_DIM,
    compute_feedback,
    encode_history,
)


class WordleEnv(gym.Env):

    metadata = {
        "render_modes": [],
    }

    def __init__(self, config):

        super().__init__()

        # ==================================================
        # Basic configuration
        # ==================================================

        self.vocabulary = list(
            config["vocabulary"]
        )

        self.targets = list(
            config["targets"]
        )

        self.max_guesses = config.get(
            "max_guesses",
            MAX_GUESSES,
        )

        self.debug = config.get(
            "debug",
            False,
        )

        # ==================================================
        # Reward configuration
        #
        # r_t =
        #   step_penalty
        #   + yellow_reward * new_yellow
        #   + green_reward * new_green
        #   + green_retention_reward * retained_green
        #   + info_gain_weight * information_gain
        #   + solve_reward * solved
        # ==================================================

        self.step_penalty = config.get(
            "step_penalty",
            -0.1,
        )

        self.yellow_reward = config.get(
            "yellow_reward",
            0.03,
        )

        self.green_reward = config.get(
            "green_reward",
            0.15,
        )

        self.green_retention_reward = (
            config.get(
                "green_retention_reward",
                0.05,
            )
        )

        self.info_gain_weight = config.get(
            "info_gain_weight",
            0.02,
        )

        self.solve_reward = config.get(
            "solve_reward",
            3.0,
        )

        # ==================================================
        # Gym spaces
        # ==================================================

        self.action_space = spaces.Discrete(
            len(self.vocabulary)
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        # ==================================================
        # Episode state
        # ==================================================

        self.target = None

        self.guesses = []
        self.feedbacks = []

        # Knowledge discovered by the agent/environment.

        self.known_letters = set()

        # position -> confirmed letter
        #
        # Example:
        # {
        #     1: "o",
        #     3: "e",
        # }

        self.known_positions = {}

        # Remaining possible answers.

        self.candidates = []

        # ==================================================
        # Diagnostic histories
        # ==================================================

        self.reward_history = []

        self.candidate_history = []

        self.info_gain_history = []

        self.yellow_history = []

        self.green_history = []

        self.green_retention_history = []

    # ======================================================
    # Reset
    # ======================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        super().reset(
            seed=seed,
        )

        # --------------------------------------------------
        # Sample target
        # --------------------------------------------------

        target_idx = self.np_random.integers(
            len(self.targets)
        )

        self.target = self.targets[
            target_idx
        ]

        # --------------------------------------------------
        # Reset Wordle history
        # --------------------------------------------------

        self.guesses = []

        self.feedbacks = []

        # --------------------------------------------------
        # Reset accumulated knowledge
        # --------------------------------------------------

        self.known_letters = set()

        self.known_positions = {}

        # --------------------------------------------------
        # Candidate solution set
        #
        # During training this contains only train targets.
        # --------------------------------------------------

        self.candidates = list(
            self.targets
        )

        # --------------------------------------------------
        # Reset diagnostics
        # --------------------------------------------------

        self.reward_history = []

        self.candidate_history = []

        self.info_gain_history = []

        self.yellow_history = []

        self.green_history = []

        self.green_retention_history = []



        observation = (
            self._get_observation()
        )

        info = {
            "target":
                self.target,

            "num_candidates":
                len(self.candidates),
        }

        return (
            observation,
            info,
        )

    # ======================================================
    # Step
    # ======================================================

    def step(
        self,
        action,
    ):

        # ==================================================
        # Action -> word
        # ==================================================

        action = int(action)

        guess = self.vocabulary[
            action
        ]

        # ==================================================
        # Green retention
        #
        # IMPORTANT:
        # Calculate this BEFORE updating known_positions.
        #
        # We are asking:
        #
        # "How many greens that were already known before
        # this action did the current guess preserve?"
        # ==================================================

        retained_green = sum(
            1
            for pos, letter
            in self.known_positions.items()
            if guess[pos] == letter
        )

        # ==================================================
        # Wordle feedback
        # ==================================================

        feedback = compute_feedback(
            guess,
            self.target,
        )

        # ==================================================
        # Candidate-set information gain
        # ==================================================

        num_candidates_before = len(
            self.candidates
        )

        new_candidates = (
            self._filter_candidates(
                guess,
                feedback,
            )
        )

        num_candidates_after = len(
            new_candidates
        )

        self.candidates = (
            new_candidates
        )

        # --------------------------------------------------
        # Information gain:
        #
        # ΔI =
        # log |C_before|
        # -
        # log |C_after|
        #
        # =
        # log(|C_before| / |C_after|)
        # --------------------------------------------------

        info_gain = (
            np.log(
                max(
                    num_candidates_before,
                    1,
                )
            )
            -
            np.log(
                max(
                    num_candidates_after,
                    1,
                )
            )
        )

        # ==================================================
        # New yellow / green information
        # ==================================================

        new_yellow = 0

        new_green = 0

        for pos, (
            letter,
            fb,
        ) in enumerate(
            zip(
                guess,
                feedback,
            )
        ):

            fb = int(fb)

            # ----------------------------------------------
            # Yellow
            # ----------------------------------------------

            if fb == 1:

                if (
                    letter
                    not in self.known_letters
                ):

                    new_yellow += 1

                self.known_letters.add(
                    letter
                )

            # ----------------------------------------------
            # Green
            # ----------------------------------------------

            elif fb == 2:

                self.known_letters.add(
                    letter
                )

                if (
                    self.known_positions.get(
                        pos
                    )
                    != letter
                ):

                    new_green += 1

                self.known_positions[
                    pos
                ] = letter

        # ==================================================
        # Save Wordle history
        # ==================================================

        self.guesses.append(
            guess
        )

        self.feedbacks.append(
            feedback
        )

        # ==================================================
        # Episode state
        # ==================================================

        solved = guess == self.target

        exhausted = len(self.guesses) >= self.max_guesses

        terminated = solved

        truncated = exhausted and not solved

        # ==================================================
        # Reward
        #
        # r_t =
        #
        # -0.05
        #
        # + 0.05 * new yellow
        #
        # + 0.15 * new green
        #
        # + 0.05 * retained green
        #
        # + 0.10 * information gain
        #
        # + 2.0 if solved
        # ==================================================

        reward = (
            self.step_penalty
            + self.yellow_reward * new_yellow
            + self.green_reward * new_green
            + self.green_retention_reward * retained_green
            + self.info_gain_weight * info_gain
            + self.solve_reward * int(solved)
        )

        reward = float(reward)

        # ==================================================
        # Diagnostics
        # ==================================================

        self.reward_history.append(reward)

        self.candidate_history.append(
            (
                num_candidates_before,
                num_candidates_after,
            )
        )

        self.info_gain_history.append(
            float(info_gain)
        )

        self.yellow_history.append(
            new_yellow
        )

        self.green_history.append(
            new_green
        )

        self.green_retention_history.append(
            retained_green
        )

        # ==================================================
        # Observation
        # ==================================================

        observation = (
            self._get_observation()
        )

        # ==================================================
        # Info
        # ==================================================

        info = {
            "target": self.target,
            "guess": guess,
            "feedback": feedback,
            "solved": int(solved),
            "num_guesses": len(self.guesses),
            "episode_reward": float(sum(self.reward_history)),
            "total_info_gain": float(sum(self.info_gain_history)),
            "total_new_yellows": int(sum(self.yellow_history)),
            "total_new_greens": int(sum(self.green_history)),
            "total_retained_greens": int(sum(self.green_retention_history)),
            "wordle_guesses": 1,
            "reward": reward,
            "candidates_before": num_candidates_before,
            "candidates_after": num_candidates_after,
        }


        if (self.debug and (terminated or truncated)):
            self._print_episode()
        
        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    
    # ======================================================
    # Observation
    # ======================================================

    def _get_observation(
        self,
    ):

        observation = encode_history(
            self.guesses,
            self.feedbacks,
            max_guesses=self.max_guesses,
        )

        return np.asarray(
            observation,
            dtype=np.float32,
        )

    # ======================================================
    # Candidate filtering
    # ======================================================

    def _filter_candidates(
        self,
        guess,
        feedback,
    ):

        observed_feedback = tuple(
            int(x)
            for x in feedback
        )

        new_candidates = []

        for candidate in (
            self.candidates
        ):

            # ----------------------------------------------
            # Ask:
            #
            # If this candidate were the true target,
            # would the current guess have produced
            # exactly the feedback that we observed?
            #
            # This automatically handles duplicate letters.
            # ----------------------------------------------

            candidate_feedback = (
                compute_feedback(
                    guess,
                    candidate,
                )
            )

            candidate_feedback = tuple(
                int(x)
                for x in candidate_feedback
            )

            if (
                candidate_feedback
                == observed_feedback
            ):

                new_candidates.append(
                    candidate
                )

        return new_candidates

    # ======================================================
    # Episode debug printer
    # ======================================================

    def _print_episode(
        self,
    ):

        symbols = {
            0: "⬛",
            1: "🟨",
            2: "🟩",
        }

        print(
            "\n"
            + "=" * 95
        )

        print(
            f"TARGET: "
            f"{self.target.upper()}"
        )

        print(
            "-" * 95
        )

        total_reward = 0.0

        for i, (
            guess,
            feedback,
            reward,
            candidate_counts,
            info_gain,
            new_yellow,
            new_green,
            retained_green,
        ) in enumerate(
            zip(
                self.guesses,
                self.feedbacks,
                self.reward_history,
                self.candidate_history,
                self.info_gain_history,
                self.yellow_history,
                self.green_history,
                self.green_retention_history,
            ),
            start=1,
        ):

            feedback_str = "".join(
                symbols[int(x)]
                for x in feedback
            )

            (
                before,
                after,
            ) = candidate_counts

            total_reward += (
                reward
            )

            print(
                f"{i:>2}. "
                f"{guess.upper():<7} "
                f"{feedback_str}  "
                f"C: "
                f"{before:>4}"
                f" -> "
                f"{after:<4}  "
                f"Y+={new_yellow}  "
                f"G+={new_green}  "
                f"GR={retained_green}  "
                f"IG={info_gain:>6.3f}  "
                f"r={reward:+.3f}"
            )

        print(
            "-" * 95
        )

        solved = (
            self.guesses
            and
            self.guesses[-1]
            == self.target
        )

        if solved:

            print(
                f"SOLVED in "
                f"{len(self.guesses)} "
                f"guesses"
            )

        else:

            print(
                f"FAILED after "
                f"{len(self.guesses)} "
                f"guesses"
            )

        print(
            f"Episode return: "
            f"{total_reward:+.3f}"
        )

        print(
            f"Remaining candidates: "
            f"{len(self.candidates)}"
        )

        print(
            "=" * 95,
            flush=True,
        )