"""
Negative sampling — the noise distribution
===========================================

For each real (center, context) pair we need K "negative" words: random words
that probably AREN'T this center's neighbour. The model learns to say "yes" to
the real pair and "no" to the negatives.

Where do negatives come from? Not uniformly. The original Word2Vec paper draws
them from the unigram (word-frequency) distribution raised to the 3/4 power:

    P(w) = count(w)^0.75 / sum_j count(j)^0.75

Intuition for the 3/4 exponent: pure frequency would drown the model in "the",
"a", "of" as negatives; uniform would over-sample rare words. The 0.75 power is
an empirical sweet spot — it dampens frequent words and lifts rare ones, so
negatives are varied but still realistic.
"""

import torch


def build_noise_distribution(token_ids: list[int], vocab_size: int,
                             power: float = 0.75) -> torch.Tensor:
    """
    Build the sampling probability for every word ID.

    Shapes:
        counts -> (V,)   raw occurrence count per word
        probs  -> (V,)   normalised, sums to 1, ready for torch.multinomial
    """
    counts = torch.zeros(vocab_size)
    for tid in token_ids:
        counts[tid] += 1

    weights = counts ** power          # (V,)  dampen frequent, lift rare
    probs = weights / weights.sum()    # (V,)  normalise to a distribution
    return probs


def draw_negatives(noise_dist: torch.Tensor, batch_size: int, k: int) -> torch.Tensor:
    """
    Draw K negative word IDs for each of the `batch_size` positive pairs.

    torch.multinomial samples indices in proportion to noise_dist. We sample
    batch_size * k of them in one call (fast) then reshape.

    Shape: () -> (batch_size, k)

    Note: we don't bother excluding the true context from the negatives. With a
    large vocab the collision chance is tiny, and the original paper ignores it
    too. Kept simple on purpose.
    """
    flat = torch.multinomial(noise_dist, batch_size * k, replacement=True)
    return flat.view(batch_size, k)
