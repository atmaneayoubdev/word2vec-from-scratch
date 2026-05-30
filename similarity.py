"""
Step 5 — Cosine similarity & nearest neighbours
===============================================

After training, the learned word vectors live in the embedding table:
    E = model.embeddings.weight    shape (V, D)
Row k is the vector for word k.

We measure word similarity with COSINE similarity — the angle between two
vectors, ignoring their length:

    cos(a, b) = (a . b) / (||a|| * ||b||)

Why cosine and not raw dot product or Euclidean distance? In embedding space
*direction* encodes meaning; magnitude mostly reflects word frequency. Cosine
strips magnitude out, so frequent and rare words compare fairly.

Trick: if we L2-normalise every row first (make each length 1), then the
cosine between any two words is just their dot product. So one normalisation
turns the whole "compare against everything" step into a single matrix-vector
product.
"""

import torch


def get_embeddings(model) -> torch.Tensor:
    """
    Pull the learned embedding matrix out of the model.

    Shape: (V, D). Detached from the autograd graph since we're only reading.
    """
    return model.embeddings.weight.detach()


def most_similar(word: str, word_to_id, id_to_word, embeddings: torch.Tensor,
                 top_k: int = 5):
    """
    Return the top_k words most similar to `word` by cosine similarity.

    Shapes:
        embeddings : (V, D)
        normed     : (V, D)   — each row scaled to unit length
        query      : (D,)     — the unit vector for `word`
        sims       : (V,)     — cosine sim of `word` vs every vocab word
    """
    if word not in word_to_id:
        raise KeyError(f"{word!r} is not in the vocabulary")

    # L2-normalise every row -> cosine becomes a plain dot product.
    # add a tiny epsilon to avoid divide-by-zero for any all-zero row.
    normed = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)  # (V, D)

    query = normed[word_to_id[word]]      # (D,)
    sims = normed @ query                 # (V, D) @ (D,) -> (V,)

    # Sort high-to-low. The first hit is the word itself (cos = 1), so skip it.
    order = torch.argsort(sims, descending=True).tolist()
    results = []
    for idx in order:
        if idx == word_to_id[word]:
            continue
        results.append((id_to_word[idx], sims[idx].item()))
        if len(results) == top_k:
            break
    return results
