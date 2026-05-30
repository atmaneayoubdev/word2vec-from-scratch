"""
Training loop for Skip-Gram with Negative Sampling
==================================================

Same skeleton as the softmax trainer, with one new step inside the loop:
for every batch of positive pairs we DRAW K negatives from the noise
distribution, then hand (center, context, negatives) to the model's loss.

There's no nn.CrossEntropyLoss here — the SGNS loss lives inside the model
because it needs all three tensors at once.
"""

import torch

from model_ns import SkipGramNS
from negative_sampling import build_noise_distribution, draw_negatives


def train_ns(token_ids, vocab_size, dataloader,
             embedding_dim: int = 16,
             epochs: int = 100,
             lr: float = 0.01,
             k_negatives: int = 5) -> SkipGramNS:
    """
    Shapes inside the loop:
        centers   : (batch,)
        contexts  : (batch,)
        negatives : (batch, K)
        loss      : scalar
    """
    model = SkipGramNS(vocab_size, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    noise_dist = build_noise_distribution(token_ids, vocab_size)  # (V,)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0

        for centers, contexts in dataloader:
            batch_size = centers.shape[0]
            negatives = draw_negatives(noise_dist, batch_size, k_negatives)  # (batch, K)

            optimizer.zero_grad()
            loss = model.loss(centers, contexts, negatives)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0 or epoch == 1:
            avg = total_loss / n_batches
            print(f"epoch {epoch:>3} | avg loss {avg:.4f}")

    return model
