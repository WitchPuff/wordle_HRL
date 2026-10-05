from ray.rllib.callbacks.callbacks import RLlibCallback


class WordleMetricsCallback(RLlibCallback):

    def on_episode_end(
        self,
        *,
        episode,
        metrics_logger,
        **kwargs,
    ):

        infos = episode.get_infos()

        # ==================================================
        # Find final Wordle info
        # ==================================================

        def find_wordle_info(obj):

            if isinstance(obj, dict):

                if (
                    "solved" in obj
                    and
                    "num_guesses" in obj
                ):
                    return obj

                for value in obj.values():

                    found = find_wordle_info(
                        value
                    )

                    if found is not None:
                        return found

            elif isinstance(
                obj,
                (list, tuple),
            ):

                for value in reversed(
                    obj
                ):

                    found = find_wordle_info(
                        value
                    )

                    if found is not None:
                        return found

            return None

        final_info = find_wordle_info(
            infos
        )

        if final_info is None:
            return

        # ==================================================
        # Episode-level metrics
        #
        # Everything has already been computed by the env.
        # Callback only aggregates across episodes.
        # ==================================================

        metrics_logger.log_value(
            "solve_rate",
            float(
                final_info["solved"]
            ),
            reduce="mean",
        )

        if final_info["solved"]:

            metrics_logger.log_value(
                "mean_guesses_solved",
                float(
                    final_info[
                        "num_guesses"
                    ]
                ),
                reduce="mean",
            )

        metrics_logger.log_value(
            "wordle_episode_reward",
            float(
                final_info[
                    "episode_reward"
                ]
            ),
            reduce="mean",
        )

        metrics_logger.log_value(
            "mean_info_gain",
            float(
                final_info[
                    "total_info_gain"
                ]
            ),
            reduce="mean",
        )

        metrics_logger.log_value(
            "mean_new_yellows",
            float(
                final_info[
                    "total_new_yellows"
                ]
            ),
            reduce="mean",
        )

        metrics_logger.log_value(
            "mean_new_greens",
            float(
                final_info[
                    "total_new_greens"
                ]
            ),
            reduce="mean",
        )

        metrics_logger.log_value(
            "mean_retained_greens",
            float(
                final_info[
                    "total_retained_greens"
                ]
            ),
            reduce="mean",
        )
        metrics_logger.log_value(
                    "mean_lost_greens",
                    float(
                        final_info[
                            "total_lost_greens"
                        ]
                    ),
                    reduce="mean",
                )
        metrics_logger.log_value(
            "mean_final_candidates",
            float(final_info["candidates_after"]),
            reduce="mean",
        )