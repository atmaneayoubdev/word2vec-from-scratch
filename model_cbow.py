"""
CBOW model
==========

Same two layers as the Skip-Gram softmax model, but the input is now a BAG of
context words instead of a single center word:

    context_ids --(Embedding)--> per-word vectors --(mean)--> one vector
                --(Linear)--> logits over the whole vocab (predict the center)

The "bag" part: we look up an embedding for each context word and AVERAGE them.
Averaging is why it's order-insensitive ("bag of words") — [the, king] and
[king, the] produce the same input vector. That averaged vector is the model's
guess at "what center word fits this neighbourhood?".
"""

import torch
import torch.nn as nn


class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        # Same embedding table idea: row k is word k's vector. (V, D)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Project the averaged context vector to a score per vocab word.
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        """
        context_ids : LongTensor (batch, C)   C context words per example

        Step A — embed every context word:
            (batch, C)  ->  (batch, C, D)
        Step B — average over the C context words (dim=1):
            (batch, C, D)  ->  (batch, D)      the bag-of-words vector
        Step C — score every vocab word:
            (batch, D)  ->  (batch, V)         raw logits (no softmax)
        """
        embedded = self.embeddings(context_ids)   # (batch, C, D)
        bag = embedded.mean(dim=1)                 # (batch, D)
        logits = self.output(bag)                  # (batch, V)
        return logits
