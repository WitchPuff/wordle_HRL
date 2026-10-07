import torch
import torch.nn as nn

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI


class HighLevelEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 256, strategy_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.strategy_head = nn.Linear(hidden_dim, strategy_dim)

    def forward(self, obs):
        h = self.encoder(obs)
        z = self.strategy_head(h)
        return h, z


class LowLevelEncoder(nn.Module):
    def __init__(self, obs_dim: int, strategy_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + strategy_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, obs, strategy):
        return self.network(torch.cat([obs, strategy], dim=-1))


class CandidateScorer(nn.Module):
    def __init__(self, state_dim: int, word_dim: int, projection_dim: int = 256):
        super().__init__()
        self.state_projection = nn.Linear(state_dim, projection_dim)
        self.word_projection = nn.Linear(word_dim, projection_dim)
        self.scale = projection_dim ** -0.5

    def forward(self, state, word_embeddings):
        state_repr = self.state_projection(state)
        word_repr = self.word_projection(word_embeddings)
        return (state_repr @ word_repr.T) * self.scale


class HierarchicalWordleRLModule(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        super().setup()
        cfg = self.model_config

        obs_dim = self.observation_space["obs"].shape[0]
        high_dim = cfg.get("high_dim", 256)
        strategy_dim = cfg.get("strategy_dim", 128)
        low_dim = cfg.get("low_dim", 256)
        projection_dim = cfg.get("projection_dim", 256)
        word_dim = cfg["word_dim"]
        embedding_path = cfg["candidate_embeddings_path"]

        self.high_encoder = HighLevelEncoder(obs_dim, high_dim, strategy_dim)
        self.low_encoder = LowLevelEncoder(obs_dim, strategy_dim, low_dim)
        self.candidate_scorer = CandidateScorer(low_dim, word_dim, projection_dim)
        self.value_head = nn.Linear(high_dim, 1)

        embeddings = torch.load(embedding_path, map_location="cpu", weights_only=True).float()
        if embeddings.shape[0] != self.action_space.n:
            raise ValueError(f"Embedding vocabulary size does not match action space: {embeddings.shape[0]} vs {self.action_space.n}")

        self.register_buffer("candidate_embeddings", embeddings)

    def encode_high(self, obs):
        return self.high_encoder(obs.float())

    def encode_low(self, obs, strategy=None):
        obs = obs.float()
        if strategy is None:
            _, strategy = self.encode_high(obs)
        return self.low_encoder(obs, strategy)

    def _forward(self, batch, **kwargs):
        observation = batch[Columns.OBS]
        obs = observation["obs"].float()
        candidate_mask = observation["candidate_mask"].float()
        soft_mask_penalty = observation["soft_mask_penalty"].float()

        high_repr, strategy = self.encode_high(obs)
        low_repr = self.encode_low(obs, strategy)

        logits = self.candidate_scorer(low_repr, self.candidate_embeddings)
        logits = logits - soft_mask_penalty * (1.0 - candidate_mask)

        return {Columns.ACTION_DIST_INPUTS: logits}

    def compute_values(self, batch, embeddings=None):
        obs = batch[Columns.OBS]["obs"].float()
        high_repr, _ = self.encode_high(obs)
        return self.value_head(high_repr).squeeze(-1)

    def get_high_representation(self, obs):
        if isinstance(obs, dict):
            obs = obs["obs"]
        high_repr, _ = self.encode_high(obs)
        return high_repr

    def get_strategy_representation(self, obs):
        if isinstance(obs, dict):
            obs = obs["obs"]
        _, strategy = self.encode_high(obs)
        return strategy

    def get_low_representation(self, obs):
        if isinstance(obs, dict):
            obs = obs["obs"]
        _, strategy = self.encode_high(obs)
        return self.encode_low(obs, strategy)

    def get_representation(self, obs):
        if isinstance(obs, dict):
            obs = obs["obs"]

        high_repr, strategy = self.encode_high(obs)
        low_repr = self.encode_low(obs, strategy)

        return {
            "high": high_repr,
            "strategy": strategy,
            "low": low_repr,
        }