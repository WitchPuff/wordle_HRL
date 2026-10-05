# models/lm.py

from typing import List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CharacterLM(nn.Module):
    """
    Frozen pretrained character-level encoder.

    Role in this project:
        allowed word -> linguistic action representation

    The LM does NOT generate Wordle guesses.
    """

    def __init__(
        self,
        model_name: str = "google/canine-s",
        freeze: bool = True,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.hidden_dim = self.model.config.hidden_size

        if freeze:
            self.freeze()

    def freeze(self):
        self.model.requires_grad_(False)
        self.model.eval()

    def forward(
        self,
        words: List[str],
        return_hidden_states: bool = False,
    ):
        device = next(self.model.parameters()).device

        inputs = self.tokenizer(
            words,
            padding=True,
            return_tensors="pt",
        ).to(device)

        outputs = self.model(
            **inputs,
            output_hidden_states=return_hidden_states,
        )

        hidden = outputs.last_hidden_state

        mask = (
            inputs["attention_mask"]
            .unsqueeze(-1)
            .to(hidden.dtype)
        )

        # Mean pooling over non-padding positions.
        embedding = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        if return_hidden_states:
            return {
                "embedding": embedding,
                "hidden_states": outputs.hidden_states,
                "attention_mask": inputs["attention_mask"],
            }

        return embedding


@torch.no_grad()
def precompute_vocabulary_embeddings(
    model: CharacterLM,
    vocabulary: List[str],
    batch_size: int = 256,
):
    """
    Encode every legal Wordle action exactly once.

    Returns:
        [vocab_size, lm_hidden_dim]
    """

    model.eval()

    all_embeddings = []

    for start in range(0, len(vocabulary), batch_size):
        batch = vocabulary[start:start + batch_size]

        embeddings = model(batch)

        all_embeddings.append(
            embeddings.cpu()
        )

    return torch.cat(
        all_embeddings,
        dim=0,
    )