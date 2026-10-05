# models/rl.py

import torch
import torch.nn as nn

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.apis.value_function_api import (
    ValueFunctionAPI,
)


class TaskEncoder(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.network(obs)


class CandidateScorer(nn.Module):
    """
    Scores every legal Wordle action using its action embedding.

        task state h_s
              |
            phi(.)
              |
              +---- dot product ----+
                                   |
        word embedding e_w         |
              |                    |
            psi(.) ----------------+

    LM embeddings themselves remain frozen.
    phi and psi are learned by RL.
    """

    def __init__(
        self,
        state_dim: int,
        word_dim: int,
        projection_dim: int = 256,
    ):
        super().__init__()

        self.state_projection = nn.Linear(
            state_dim,
            projection_dim,
        )

        self.word_projection = nn.Linear(
            word_dim,
            projection_dim,
        )

        self.scale = projection_dim ** -0.5

    def forward(
        self,
        state,
        word_embeddings,
    ):
        state_repr = self.state_projection(state)

        word_repr = self.word_projection(
            word_embeddings
        )

        return (
            state_repr @ word_repr.T
        ) * self.scale


class FlatWordleRLModule(
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
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
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
                "Embedding vocabulary size does not match "
                f"action space: {embeddings.shape[0]} "
                f"vs {self.action_space.n}"
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
            obs = batch[Columns.OBS].float()
            embeddings = self.encode_state(obs)

        return self.value_head(
            embeddings
        ).squeeze(-1)

    # Later used by probing.
    def get_representation(self, obs):
        return self.encode_state(obs)