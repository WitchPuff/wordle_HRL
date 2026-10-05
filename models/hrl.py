# models/hrl.py

import torch
import torch.nn as nn

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.apis.value_function_api import (
    ValueFunctionAPI,
)

from models.rl import (
    TaskEncoder,
    CandidateScorer,
)


class WordleHighLevelModule(
    TorchRLModule,
    ValueFunctionAPI,
):

    def setup(self):
        super().setup()

        cfg = self.model_config

        obs_dim = self.observation_space.shape[0]

        hidden_dim = cfg.get(
            "hidden_dim",
            256,
        )

        num_options = self.action_space.n

        self.task_encoder = TaskEncoder(
            obs_dim,
            hidden_dim,
        )

        self.option_head = nn.Linear(
            hidden_dim,
            num_options,
        )

        self.value_head = nn.Linear(
            hidden_dim,
            1,
        )

    def encode_state(self, obs):
        return self.task_encoder(
            obs.float()
        )

    def _forward(
        self,
        batch,
        **kwargs,
    ):
        obs = batch[Columns.OBS].float()

        h = self.encode_state(obs)

        logits = self.option_head(h)

        return {
            Columns.ACTION_DIST_INPUTS: logits,
        }

    def compute_values(
        self,
        batch,
        embeddings=None,
    ):
        if embeddings is None:
            embeddings = self.encode_state(
                batch[Columns.OBS]
            )

        return self.value_head(
            embeddings
        ).squeeze(-1)

    def get_representation(self, obs):
        return self.encode_state(obs)


class WordleLowLevelModule(
    TorchRLModule,
    ValueFunctionAPI,
):

    def setup(self):
        super().setup()

        cfg = self.model_config

        obs_dim = self.observation_space.shape[0]

        hidden_dim = cfg.get(
            "hidden_dim",
            256,
        )

        projection_dim = cfg.get(
            "projection_dim",
            256,
        )

        word_dim = cfg["word_dim"]

        embedding_path = cfg[
            "candidate_embeddings_path"
        ]

        self.task_encoder = TaskEncoder(
            obs_dim,
            hidden_dim,
        )

        self.candidate_scorer = CandidateScorer(
            state_dim=hidden_dim,
            word_dim=word_dim,
            projection_dim=projection_dim,
        )

        self.value_head = nn.Linear(
            hidden_dim,
            1,
        )

        embeddings = torch.load(
            embedding_path,
            map_location="cpu",
            weights_only=True,
        ).float()

        if embeddings.shape[0] != self.action_space.n:
            raise ValueError(
                "Embedding vocabulary size does not "
                "match low-level action space."
            )

        self.register_buffer(
            "candidate_embeddings",
            embeddings,
        )

    def encode_state(self, obs):
        return self.task_encoder(
            obs.float()
        )

    def _forward(
        self,
        batch,
        **kwargs,
    ):
        obs = batch[Columns.OBS].float()

        h = self.encode_state(obs)

        logits = self.candidate_scorer(
            h,
            self.candidate_embeddings,
        )

        return {
            Columns.ACTION_DIST_INPUTS: logits,
        }

    def compute_values(
        self,
        batch,
        embeddings=None,
    ):
        if embeddings is None:
            embeddings = self.encode_state(
                batch[Columns.OBS]
            )

        return self.value_head(
            embeddings
        ).squeeze(-1)

    def get_representation(self, obs):
        return self.encode_state(obs)