"""
Skip-Gram with Negative Sampling (SGNS) — the model
====================================================

The big structural change vs. the softmax version: there is NO Linear(D, V)
output layer and NO softmax. Instead we keep TWO embedding tables:

    in_embeddings  (V, D) : the word as a CENTER word   (these are the vectors
                            we keep at the end)
    out_embeddings (V, D) : the word as a CONTEXT word

A pair's score is just the dot product of the center's input vector and the
context's output vector:  score = v_center . u_context

Why two tables? A word plays two different roles ("center" vs "context"), and
letting it have a separate vector per role makes the optimisation cleaner. By
convention we throw away out_embeddings at the end and keep in_embeddings.

The loss (this is the whole idea):

    L = -log σ(v_c . u_pos)  -  Σ_neg log σ(-v_c . u_neg)

    - First term: push the REAL pair's dot product UP   (σ -> 1, "yes").
    - Second term: push each NEGATIVE pair's dot product DOWN (σ -> 0, "no").

σ is the logistic sigmoid, so each term is a binary classification. We've
replaced one V-way softmax with (1 + K) cheap yes/no decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramNS(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def loss(self, center: torch.Tensor, context: torch.Tensor,
             negatives: torch.Tensor) -> torch.Tensor:
        """
        center    : (batch,)        center word IDs
        context   : (batch,)        true context IDs (the positives)
        negatives : (batch, K)      sampled noise IDs

        Returns a scalar loss averaged over the batch.
        """
        v_c = self.in_embeddings(center)        # (batch, D)
        u_pos = self.out_embeddings(context)    # (batch, D)
        u_neg = self.out_embeddings(negatives)  # (batch, K, D)

        # --- positive term -------------------------------------------------
        # elementwise multiply then sum over D == dot product per row.
        pos_score = (v_c * u_pos).sum(dim=1)            # (batch,)
        pos_loss = F.logsigmoid(pos_score)             # (batch,)  log σ(+score)

        # --- negative term -------------------------------------------------
        # batched matrix-vector product: for each row, dot v_c with each of its
        # K negative vectors. bmm needs (batch, K, D) x (batch, D, 1).
        neg_score = torch.bmm(u_neg, v_c.unsqueeze(2)).squeeze(2)  # (batch, K)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)            # (batch,)  Σ log σ(-score)

        # We maximise (pos_loss + neg_loss), i.e. minimise its negative.
        return -(pos_loss + neg_loss).mean()
