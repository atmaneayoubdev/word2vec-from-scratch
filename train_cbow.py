"""
CBOW training loop
==================

Identical in spirit to the Skip-Gram softmax trainer — predict one word with
CrossEntropyLoss — the only difference is the input is a (batch, C) bag of
context words instead of a single (batch,) center.
"""

import torch
import torch.nn as nn

from model_cbow import CBOW


def train_cbow(vocab_size, dataloader,
               embedding_dim: int = 16,
               epochs: int = 100,
               lr: float = 0.01) -> CBOW:
    """
    Shapes inside the loop:
        contexts : (batch, C)    bag of context word IDs (the input)
        centers  : (batch,)      center word to predict   (the label)
        logits   : (batch, V)
        loss     : scalar
    """
    model = CBOW(vocab_size, embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0

        for contexts, centers in dataloader:
            optimizer.zero_grad()
            logits = model(contexts)          # (batch, V)
            loss = criterion(logits, centers)  # predict the center word
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0 or epoch == 1:
            avg = total_loss / n_batches
            print(f"epoch {epoch:>3} | avg loss {avg:.4f}")

    return model
