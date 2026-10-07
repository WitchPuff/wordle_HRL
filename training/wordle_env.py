import numpy as np
import gymnasium as gym
from gymnasium import spaces

from training.wordle_core import MAX_GUESSES, OBS_DIM, compute_feedback, encode_history


class WordleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config):
        super().__init__()

        self.vocabulary = list(config["vocabulary"])
        self.targets = list(config["targets"])
        self.candidate_targets = list(config.get("candidate_targets", self.targets))
        self.word_to_idx = {word: i for i, word in enumerate(self.vocabulary)}

        self.max_guesses = config.get("max_guesses", MAX_GUESSES)
        self.debug = config.get("debug", False)

        self.step_penalty = config.get("step_penalty", -0.05)
        self.yellow_reward = config.get("yellow_reward", 0.03)
        self.green_reward = config.get("green_reward", 0.15)
        self.info_gain_weight = config.get("info_gain_weight", 0.005)
        self.solve_reward = config.get("solve_reward", 10.0)
        self.soft_mask_penalty = float(config.get("soft_mask_penalty", 0.0))

        self.action_space = spaces.Discrete(len(self.vocabulary))
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32),
            "candidate_mask": spaces.Box(
                0.0,
                1.0,
                shape=(len(self.vocabulary),),
                dtype=np.float32,
            ),
            "soft_mask_penalty": spaces.Box(
                0.0,
                np.inf,
                shape=(1,),
                dtype=np.float32,
            ),
        })

        self.target = None
        self.guesses = []
        self.feedbacks = []
        self.known_letters = set()
        self.known_positions = {}
        self.candidates = []

        self.reward_history = []
        self.candidate_history = []
        self.info_gain_history = []
        self.yellow_history = []
        self.green_history = []
        self.green_retention_history = []
        self.green_loss_history = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if options and "target" in options:
            self.target = options["target"]
        else:
            self.target = self.targets[self.np_random.integers(len(self.targets))]

        self.guesses = []
        self.feedbacks = []
        self.known_letters = set()
        self.known_positions = {}
        self.candidates = list(self.candidate_targets)

        self.reward_history = []
        self.candidate_history = []
        self.info_gain_history = []
        self.yellow_history = []
        self.green_history = []
        self.green_retention_history = []
        self.green_loss_history = []

        return self._get_observation(), {
            "target": self.target,
            "num_candidates": len(self.candidates),
        }

    def step(self, action):
        guess = self.vocabulary[int(action)]

        retained_green = sum(
            guess[pos] == letter
            for pos, letter in self.known_positions.items()
        )
        lost_green = len(self.known_positions) - retained_green

        feedback = compute_feedback(guess, self.target)

        num_candidates_before = len(self.candidates)
        self.candidates = self._filter_candidates(guess, feedback)
        num_candidates_after = len(self.candidates)

        info_gain = (
            np.log(max(num_candidates_before, 1))
            - np.log(max(num_candidates_after, 1))
        )

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

        self.guesses.append(guess)
        self.feedbacks.append(feedback)

        solved = guess == self.target
        exhausted = len(self.guesses) >= self.max_guesses

        terminated = solved
        truncated = exhausted and not solved

        reward = float(
            self.step_penalty
            + self.yellow_reward * new_yellow
            + self.green_reward * new_green
            + self.info_gain_weight * info_gain
            + self.solve_reward * int(solved)
        )

        self.reward_history.append(reward)
        self.candidate_history.append(
            (num_candidates_before, num_candidates_after)
        )
        self.info_gain_history.append(float(info_gain))
        self.yellow_history.append(new_yellow)
        self.green_history.append(new_green)
        self.green_retention_history.append(retained_green)
        self.green_loss_history.append(lost_green)

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
            "num_candidates": len(self.candidates),
        }

        if self.debug and (terminated or truncated):
            self._print_episode()

        return observation, reward, terminated, truncated, info

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
            "candidate_mask": self._get_candidate_mask(),
            "soft_mask_penalty": np.asarray(
                [self.soft_mask_penalty],
                dtype=np.float32,
            ),
        }

    def _get_candidate_mask(self):
        mask = np.zeros(
            len(self.vocabulary),
            dtype=np.float32,
        )

        for word in self.candidates:
            idx = self.word_to_idx.get(word)

            if idx is not None:
                mask[idx] = 1.0

        return mask

    def _filter_candidates(self, guess, feedback):
        observed = tuple(int(x) for x in feedback)

        return [
            candidate
            for candidate in self.candidates
            if tuple(
                int(x)
                for x in compute_feedback(guess, candidate)
            ) == observed
        ]

    def _print_episode(self):
        symbols = {
            0: "⬛",
            1: "🟨",
            2: "🟩",
        }

        print("\n" + "=" * 95)
        print(f"TARGET: {self.target.upper()}")
        print("-" * 95)

        for i, (
            guess,
            feedback,
            reward,
            counts,
            info_gain,
            yellow,
            green,
            retained,
            lost,
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
            1,
        ):
            feedback_str = "".join(
                symbols[int(x)]
                for x in feedback
            )

            before, after = counts

            print(
                f"{i:>2}. {guess.upper():<7} {feedback_str}  "
                f"C: {before:>4} -> {after:<4}  "
                f"Y+={yellow}  G+={green}  "
                f"GR={retained}  G-={lost}  "
                f"IG={info_gain:>6.3f}  "
                f"r={reward:+.3f}"
            )

        solved = bool(
            self.guesses
            and self.guesses[-1] == self.target
        )

        print("-" * 95)
        print(
            f"{'SOLVED' if solved else 'FAILED'} "
            f"after {len(self.guesses)} guesses"
        )
        print(
            f"Episode return: "
            f"{sum(self.reward_history):+.3f}"
        )
        print(
            f"Remaining candidates: "
            f"{len(self.candidates)}"
        )
        print("=" * 95, flush=True)