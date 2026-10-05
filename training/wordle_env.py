import numpy as np
import gymnasium as gym
from gymnasium import spaces

from training.wordle_core import MAX_GUESSES, OBS_DIM, compute_feedback, encode_history


class WordleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config):
        super().__init__()

        # ==================================================
        # Configuration
        # ==================================================

        self.vocabulary = list(config["vocabulary"])
        self.targets = list(config["targets"])
        self.word_to_idx = {word: i for i, word in enumerate(self.vocabulary)}

        self.max_guesses = config.get("max_guesses", MAX_GUESSES)
        self.debug = config.get("debug", False)

        # ==================================================
        # Reward
        # ==================================================

        self.step_penalty = config.get("step_penalty", -0.05)
        self.yellow_reward = config.get("yellow_reward", 0.03)
        self.green_reward = config.get("green_reward", 0.15)
        self.info_gain_weight = config.get("info_gain_weight", 0.005)
        self.solve_reward = config.get("solve_reward", 10.0)

        # ==================================================
        # Gym spaces
        # ==================================================

        self.action_space = spaces.Discrete(len(self.vocabulary))

        self.observation_space = spaces.Dict({
            "obs": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(OBS_DIM,),
                dtype=np.float32,
            ),
            "action_mask": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(len(self.vocabulary),),
                dtype=np.float32,
            ),
        })

        # ==================================================
        # Episode state
        # ==================================================

        self.target = None
        self.guesses = []
        self.feedbacks = []

        self.known_letters = set()
        self.known_positions = {}
        self.candidates = []

        # ==================================================
        # Diagnostics
        # ==================================================

        self.reward_history = []
        self.candidate_history = []
        self.info_gain_history = []
        self.yellow_history = []
        self.green_history = []
        self.green_retention_history = []
        self.green_loss_history = []

    # ======================================================
    # Reset
    # ======================================================

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.target = self.targets[self.np_random.integers(len(self.targets))]

        self.guesses = []
        self.feedbacks = []

        self.known_letters = set()
        self.known_positions = {}

        # Initially every training target is possible.
        self.candidates = list(self.targets)

        self.reward_history = []
        self.candidate_history = []
        self.info_gain_history = []
        self.yellow_history = []
        self.green_history = []
        self.green_retention_history = []
        self.green_loss_history = []

        observation = self._get_observation()

        info = {
            "target": self.target,
            "num_candidates": len(self.candidates),
        }

        return observation, info

    # ======================================================
    # Step
    # ======================================================

    def step(self, action):
        action = int(action)
        guess = self.vocabulary[action]

        # --------------------------------------------------
        # Previously known greens
        # --------------------------------------------------

        retained_green = sum(
            1
            for pos, letter in self.known_positions.items()
            if guess[pos] == letter
        )

        lost_green = len(self.known_positions) - retained_green

        # --------------------------------------------------
        # Feedback
        # --------------------------------------------------

        feedback = compute_feedback(guess, self.target)

        # --------------------------------------------------
        # Candidate filtering / information gain
        # --------------------------------------------------

        num_candidates_before = len(self.candidates)
        self.candidates = self._filter_candidates(guess, feedback)
        num_candidates_after = len(self.candidates)

        info_gain = (
            np.log(max(num_candidates_before, 1))
            - np.log(max(num_candidates_after, 1))
        )

        # --------------------------------------------------
        # New information
        # --------------------------------------------------

        new_yellow = 0
        new_green = 0

        for pos, (letter, fb) in enumerate(zip(guess, feedback)):
            fb = int(fb)

            if fb == 1:
                if letter not in self.known_letters:
                    new_yellow += 1
                self.known_letters.add(letter)

            elif fb == 2:
                self.known_letters.add(letter)

                if self.known_positions.get(pos) != letter:
                    new_green += 1

                self.known_positions[pos] = letter

        # --------------------------------------------------
        # Save history
        # --------------------------------------------------

        self.guesses.append(guess)
        self.feedbacks.append(feedback)

        # --------------------------------------------------
        # Episode state
        # --------------------------------------------------

        solved = guess == self.target
        exhausted = len(self.guesses) >= self.max_guesses

        terminated = solved
        truncated = exhausted and not solved

        # --------------------------------------------------
        # Reward
        # --------------------------------------------------

        reward = (
            self.step_penalty
            + self.yellow_reward * new_yellow
            + self.green_reward * new_green
            + self.info_gain_weight * info_gain
            + self.solve_reward * int(solved)
        )

        reward = float(reward)

        # --------------------------------------------------
        # Diagnostics
        # --------------------------------------------------

        self.reward_history.append(reward)
        self.candidate_history.append((num_candidates_before, num_candidates_after))
        self.info_gain_history.append(float(info_gain))
        self.yellow_history.append(new_yellow)
        self.green_history.append(new_green)
        self.green_retention_history.append(retained_green)
        self.green_loss_history.append(lost_green)

        # --------------------------------------------------
        # Next observation
        # --------------------------------------------------

        observation = self._get_observation()

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
            "total_lost_greens": int(sum(self.green_loss_history)),
            "wordle_guesses": 1,
            "reward": reward,
            "candidates_before": num_candidates_before,
            "candidates_after": num_candidates_after,
            "num_valid_actions": int(self._get_action_mask().sum()),
        }

        if self.debug and (terminated or truncated):
            self._print_episode()

        return observation, reward, terminated, truncated, info

    # ======================================================
    # Observation
    # ======================================================

    def _get_observation(self):
        obs = np.asarray(
            encode_history(
                self.guesses,
                self.feedbacks,
                max_guesses=self.max_guesses,
            ),
            dtype=np.float32,
        )

        return {
            "obs": obs,
            "action_mask": self._get_action_mask(),
        }

    # ======================================================
    # Action mask
    # ======================================================

    def _get_action_mask(self):
        mask = np.zeros(len(self.vocabulary), dtype=np.float32)

        for word in self.candidates:
            idx = self.word_to_idx.get(word)
            if idx is not None:
                mask[idx] = 1.0

        # Safety fallback: never return an all-zero mask.
        if not mask.any():
            mask[:] = 1.0

        return mask

    # ======================================================
    # Candidate filtering
    # ======================================================

    def _filter_candidates(self, guess, feedback):
        observed_feedback = tuple(int(x) for x in feedback)
        new_candidates = []

        for candidate in self.candidates:
            candidate_feedback = compute_feedback(guess, candidate)
            candidate_feedback = tuple(int(x) for x in candidate_feedback)

            if candidate_feedback == observed_feedback:
                new_candidates.append(candidate)

        return new_candidates

    # ======================================================
    # Debug
    # ======================================================

    def _print_episode(self):
        symbols = {
            0: "⬛",
            1: "🟨",
            2: "🟩",
        }

        print("\n" + "=" * 95)
        print(f"TARGET: {self.target.upper()}")
        print("-" * 95)

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
            lost_green,
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
                self.green_loss_history,
            ),
            start=1,
        ):
            feedback_str = "".join(symbols[int(x)] for x in feedback)
            before, after = candidate_counts
            total_reward += reward

            print(
                f"{i:>2}. "
                f"{guess.upper():<7} "
                f"{feedback_str}  "
                f"C: {before:>4} -> {after:<4}  "
                f"Y+={new_yellow}  "
                f"G+={new_green}  "
                f"GR={retained_green}  "
                f"G-={lost_green}  "
                f"IG={info_gain:>6.3f}  "
                f"r={reward:+.3f}"
            )

        print("-" * 95)

        solved = bool(self.guesses and self.guesses[-1] == self.target)

        if solved:
            print(f"SOLVED in {len(self.guesses)} guesses")
        else:
            print(f"FAILED after {len(self.guesses)} guesses")

        print(f"Episode return: {total_reward:+.3f}")
        print(f"Remaining candidates: {len(self.candidates)}")
        print("=" * 95, flush=True)