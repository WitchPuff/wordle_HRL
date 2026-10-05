import numpy as np

from gymnasium import spaces

from ray.rllib.env.multi_agent_env import (
    MultiAgentEnv,
)

from training.wordle_core import (
    compute_feedback,
    encode_history,
    MAX_GUESSES,
    OBS_DIM,
)


HIGH = "high_level"
LOW = "low_level"


class HierarchicalWordleEnv(
    MultiAgentEnv
):

    def __init__(
        self,
        config=None,
    ):

        super().__init__()

        config = config or {}

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

        self.num_options = config.get(
            "num_options",
            3,
        )

        self.debug = config.get(
            "debug",
            False,
        )

        if (
            self.max_guesses
            != MAX_GUESSES
        ):
            raise ValueError(
                "Current encoder assumes "
                f"max_guesses="
                f"{MAX_GUESSES}."
            )

        # ==================================================
        # Reward configuration
        #
        # Same reward as Flat Wordle.
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
            -0.05,
        )

        self.yellow_reward = config.get(
            "yellow_reward",
            0.05,
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
            0.10,
        )

        self.solve_reward = config.get(
            "solve_reward",
            2.0,
        )

        # ==================================================
        # Agents
        # ==================================================

        self.possible_agents = [
            HIGH,
            LOW,
        ]

        self.agents = (
            self.possible_agents.copy()
        )

        # ==================================================
        # Observation spaces
        # ==================================================

        self.observation_spaces = {

            HIGH:
                spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(OBS_DIM,),
                    dtype=np.float32,
                ),

            LOW:
                spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(
                        OBS_DIM
                        + self.num_options,
                    ),
                    dtype=np.float32,
                ),
        }

        # ==================================================
        # Action spaces
        # ==================================================

        self.action_spaces = {

            HIGH:
                spaces.Discrete(
                    self.num_options
                ),

            LOW:
                spaces.Discrete(
                    len(
                        self.vocabulary
                    )
                ),
        }

        # ==================================================
        # HRL state
        # ==================================================

        self.phase = HIGH

        self.current_option = None

        # ==================================================
        # Wordle state
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

        self.option_history = []

        self.reward_history = []

        self.candidate_history = []

        self.info_gain_history = []

        self.yellow_history = []

        self.green_history = []

        self.green_retention_history = []

    # ======================================================
    # RLlib spaces
    # ======================================================

    def get_observation_space(
        self,
        agent_id,
    ):

        return (
            self.observation_spaces[
                agent_id
            ]
        )

    def get_action_space(
        self,
        agent_id,
    ):

        return (
            self.action_spaces[
                agent_id
            ]
        )

    # ======================================================
    # State
    # ======================================================

    def _state(
        self,
    ):

        return np.asarray(
            encode_history(
                self.guesses,
                self.feedbacks,
                self.max_guesses,
            ),
            dtype=np.float32,
        )

    # ======================================================
    # Low-level observation
    # ======================================================

    def _low_obs(
        self,
    ):

        option_vector = np.zeros(
            self.num_options,
            dtype=np.float32,
        )

        option_vector[
            self.current_option
        ] = 1.0

        return np.concatenate(
            [
                self._state(),
                option_vector,
            ]
        ).astype(
            np.float32
        )

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
            options=options,
        )

        self.agents = (
            self.possible_agents.copy()
        )

        # --------------------------------------------------
        # Sample target
        # --------------------------------------------------

        target_idx = (
            self.np_random.integers(
                len(self.targets)
            )
        )

        self.target = (
            self.targets[
                target_idx
            ]
        )

        # --------------------------------------------------
        # Reset Wordle state
        # --------------------------------------------------

        self.guesses = []

        self.feedbacks = []

        self.known_letters = set()

        self.known_positions = {}

        self.candidates = list(
            self.targets
        )

        # --------------------------------------------------
        # Reset HRL state
        # --------------------------------------------------

        self.current_option = None

        self.phase = HIGH

        # --------------------------------------------------
        # Reset diagnostics
        # --------------------------------------------------

        self.option_history = []

        self.reward_history = []

        self.candidate_history = []

        self.info_gain_history = []

        self.yellow_history = []

        self.green_history = []

        self.green_retention_history = []

        # --------------------------------------------------
        # First decision belongs to high level
        # --------------------------------------------------

        return (
            {
                HIGH:
                    self._state()
            },
            {
                HIGH: {}
            },
        )

    # ======================================================
    # Step
    # ======================================================

    def step(
        self,
        action_dict,
    ):

        # ==================================================
        # HIGH-LEVEL DECISION
        #
        # High level chooses latent option.
        # No Wordle guess happens here.
        # ==================================================

        if (
            self.phase
            == HIGH
        ):

            self.current_option = int(
                action_dict[
                    HIGH
                ]
            )

            self.phase = LOW

            return (
                {
                    LOW:
                        self._low_obs()
                },

                {
                    HIGH:
                        0.0,
                },

                {
                    "__all__":
                        False,
                },

                {
                    "__all__":
                        False,
                },

                {
                    LOW: {
                        "option":
                            self.current_option
                    }
                },
            )

        # ==================================================
        # LOW-LEVEL DECISION
        #
        # Low level chooses actual Wordle word.
        # ==================================================

        action = int(
            action_dict[
                LOW
            ]
        )

        guess = (
            self.vocabulary[
                action
            ]
        )

        # ==================================================
        # Green retention
        #
        # Must be calculated BEFORE updating
        # known_positions.
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

        feedback = (
            compute_feedback(
                guess,
                self.target,
            )
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

            fb = int(
                fb
            )

            # ----------------------------------------------
            # Yellow
            # ----------------------------------------------

            if fb == 1:

                if (
                    letter
                    not in
                    self.known_letters
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
        # Store Wordle + HRL history
        # ==================================================

        self.guesses.append(
            guess
        )

        self.feedbacks.append(
            feedback
        )

        self.option_history.append(
            self.current_option
        )

        # ==================================================
        # Episode state
        # ==================================================

        solved = (
            guess
            == self.target
        )

        exhausted = (
            len(self.guesses)
            >= self.max_guesses
        )

        terminated = solved

        truncated = (
            exhausted
            and not solved
        )

        # ==================================================
        # Reward
        #
        # Same external reward is given to both levels.
        # ==================================================

        reward = (
            self.step_penalty
            + self.yellow_reward
            * new_yellow
            + self.green_reward
            * new_green
            + self.green_retention_reward
            * retained_green
            + self.info_gain_weight
            * info_gain
        )

        if solved:

            reward += (
                self.solve_reward
            )

        reward = float(
            reward
        )

        # ==================================================
        # Diagnostics
        # ==================================================

        self.reward_history.append(
            reward
        )

        self.candidate_history.append(
            (
                num_candidates_before,
                num_candidates_after,
            )
        )

        self.info_gain_history.append(
            float(
                info_gain
            )
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
        # Rewards
        # ==================================================

        rewards = {
            HIGH:
                reward,

            LOW:
                reward,
        }

        # ==================================================
        # Info
        # ==================================================
        high_info = {
            "option": int(self.current_option),
            "reward": reward[HIGH],
        }



        low_info = {
            "reward": reward[LOW],
            "guess":
                guess,

            "feedback":
                feedback.copy(),

            "solved":
                bool(solved),

            "num_guesses":
                len(self.guesses),

            "wordle_guesses":
                1,


            "new_yellow":
                new_yellow,

            "new_green":
                new_green,

            "retained_green":
                retained_green,

            "info_gain":
                float(
                    info_gain
                ),

            "candidates_before":
                num_candidates_before,

            "candidates_after":
                num_candidates_after,
        }

        infos = {
            HIGH: high_info,
            LOW: low_info,
        }

        # ==================================================
        # Episode finished
        # ==================================================

        if (
            terminated
            or truncated
        ):

            if self.debug:
                self._print_episode()

            # --------------------------------------------------
            # Final observations
            #
            # RLlib requires a final observation for agents that
            # are truncated so that PPO can bootstrap the value
            # function from the final state.
            # --------------------------------------------------

            final_state = (
                self._state()
            )

            final_low_obs = (
                self._low_obs()
            )

            final_observations = {
                HIGH:
                    final_state,

                LOW:
                    final_low_obs,
            }

            # --------------------------------------------------
            # Per-agent termination/truncation flags
            # --------------------------------------------------

            terminateds = {
                HIGH:
                    terminated,

                LOW:
                    terminated,

                "__all__":
                    terminated,
            }

            truncateds = {
                HIGH:
                    truncated,

                LOW:
                    truncated,

                "__all__":
                    truncated,
            }

            self.agents = []

            return (
                final_observations,
                rewards,
                terminateds,
                truncateds,
                infos,
            )

        # ==================================================
        # Next high-level decision
        # ==================================================

        self.phase = HIGH

        return (
            {
                HIGH:
                    self._state()
            },

            rewards,

            {
                "__all__":
                    False,
            },

            {
                "__all__":
                    False,
            },

            infos,
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

            candidate_feedback = (
                compute_feedback(
                    guess,
                    candidate,
                )
            )

            candidate_feedback = tuple(
                int(x)
                for x in
                candidate_feedback
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
            + "=" * 105
        )

        print(
            f"TARGET: "
            f"{self.target.upper()}"
        )

        print(
            "-" * 105
        )

        total_reward = 0.0

        for i, (
            option,
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
                self.option_history,
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
                f"[O={option}] "
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
            "-" * 105
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

        # --------------------------------------------------
        # Option usage
        # --------------------------------------------------

        option_counts = {
            option: (
                self.option_history.count(
                    option
                )
            )
            for option in range(
                self.num_options
            )
        }

        print(
            f"Option usage: "
            f"{option_counts}"
        )

        print(
            "=" * 105,
            flush=True,
        )