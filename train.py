"""
Step 4 — Training loop
======================

We optimise one objective: given a center word, raise the score of the words
that actually appeared in its context.

Loss: nn.CrossEntropyLoss
    Input  : logits  shape (batch, V)   — raw scores per vocab word
    Target : context shape (batch,)     — the correct context word ID
    It does log-softmax over the V scores, then penalises the model by how
    little probability mass it put on the true context word. Minimising it
    pushes embeddings of words that share contexts closer together.

Optimiser: Adam — a sensible, low-fuss default for this scale.
"""

import torch
import torch.nn as nn

from model import SkipGram


def train(token_ids, vocab_size, dataloader,
          embedding_dim: int = 16,
          epochs: int = 100,
          lr: float = 0.01) -> SkipGram:
    """
    Train a SkipGram model and return it.

    Shapes inside the loop:
        centers  : (batch,)        integer center IDs   (the input)
        contexts : (batch,)        integer context IDs  (the label)
        logits   : (batch, V)      raw scores from the model
        loss     : scalar          averaged over the batch
    """
    model = SkipGram(vocab_size, embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0

        for centers, contexts in dataloader:
            optimizer.zero_grad()       # clear gradients from the previous step
            logits = model(centers)     # (batch, V) forward pass
            loss = criterion(logits, contexts)  # compare scores to true context
            loss.backward()             # backprop: d(loss)/d(every parameter)
            optimizer.step()            # nudge parameters downhill

            total_loss += loss.item()
            n_batches += 1

        # Print every 10 epochs to keep the log readable.
        if epoch % 10 == 0 or epoch == 1:
            avg = total_loss / n_batches
            print(f"epoch {epoch:>3} | avg loss {avg:.4f}")

    return model
