"""
Step 3 — The Skip-Gram model
=============================

The model is tiny: two layers.

    center_id  --(Embedding)-->  v  --(Linear)-->  logits over the whole vocab

Layer 1: nn.Embedding(V, D)
    A lookup table with V rows and D columns. Row k is the D-dimensional
    embedding of word k. These rows ARE the word vectors we ultimately want.

Layer 2: nn.Linear(D, V)
    Projects the D-dim center embedding to V scores ("logits"), one per
    vocabulary word — how likely is each word to be a context of the center.

Why no one-hot vectors?
-----------------------
The textbook draws Skip-Gram as:  one_hot(center) @ W   where one_hot is
length V and W is (V, D). But multiplying a one-hot row vector by a matrix
just *selects one row* of that matrix. nn.Embedding does exactly that
selection by integer index — same math, but O(1) lookup instead of an
O(V*D) matmul against mostly zeros. That's why our inputs are plain integer
IDs, not (batch, V) one-hot tensors.
"""

import torch
import torch.nn as nn


class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Input embedding table.
        # Weight shape: (V, D). Row k = vector for word k.
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Output projection: D -> V.
        # Internally weight shape: (V, D), bias shape: (V,).
        # Each output unit scores "how context-y is word j for this center".
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_ids: torch.Tensor) -> torch.Tensor:
        """
        center_ids : LongTensor of shape (batch,)   — integer word IDs

        Step A — look up embeddings:
            v = self.embeddings(center_ids)
            shape (batch,)  ->  (batch, D)
            We pulled one D-dim row per center word.

        Step B — score every vocab word:
            logits = self.output(v)
            shape (batch, D)  ->  (batch, V)
            Row i holds V raw scores for the i-th center in the batch.

        We return RAW logits (no softmax). nn.CrossEntropyLoss applies
        log-softmax internally and is more numerically stable that way.
        """
        v = self.embeddings(center_ids)   # (batch, D)
        logits = self.output(v)           # (batch, V)
        return logits
