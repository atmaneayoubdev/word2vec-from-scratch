"""
plot.py — visualise the learned word vectors in 2-D
===================================================

Our embeddings live in D=16 dimensions, which we can't see. We project them
down to 2-D with PCA (Principal Component Analysis) and scatter them.

PCA in one sentence: rotate the cloud of points so the first two axes capture
as much of its spread (variance) as possible, then keep just those two axes.
We do it with an SVD on the mean-centred matrix — no extra libraries needed.

If the model learned anything, semantically related words (the royalty cluster
vs the pet cluster) should land near each other on the plot.

Usage:
    uv run python plot.py            # default: skip-gram
    uv run python plot.py ns         # negative sampling
    uv run python plot.py cbow       # CBOW
"""

import sys

import torch
import matplotlib.pyplot as plt

from preprocess import TINY_CORPUS, tokenize, build_vocab, encode

# Two loose semantic groups, just for colouring the scatter so the structure
# pops visually. These labels are OURS — the model never sees them.
ROYALTY = {"king", "queen", "kingdom", "man", "woman", "rules",
           "becomes", "beside", "loves"}
PETS = {"cat", "dog", "pet", "sleeps", "near", "small", "loyal", "chases"}


def get_embeddings(variant: str):
    """
    Train the chosen variant and return (embeddings (V,D), word_to_id, id_to_word).
    """
    torch.manual_seed(0)
    tokens = tokenize(TINY_CORPUS)
    word_to_id, id_to_word = build_vocab(tokens)
    token_ids = encode(tokens, word_to_id)
    vocab_size = len(word_to_id)

    if variant == "cbow":
        from dataset_cbow import make_cbow_dataloader
        from train_cbow import train_cbow
        loader = make_cbow_dataloader(token_ids, window_size=2, batch_size=16)
        model = train_cbow(vocab_size, loader, embedding_dim=16, epochs=100)
        emb = model.embeddings.weight.detach()

    elif variant == "ns":
        from dataset import make_dataloader
        from train_ns import train_ns
        loader = make_dataloader(token_ids, window_size=2, batch_size=16)
        model = train_ns(token_ids, vocab_size, loader, embedding_dim=16, epochs=100)
        emb = model.in_embeddings.weight.detach()

    else:  # "skipgram"
        from dataset import make_dataloader
        from train import train
        loader = make_dataloader(token_ids, window_size=2, batch_size=16)
        model = train(token_ids, vocab_size, loader, embedding_dim=16, epochs=100)
        emb = model.embeddings.weight.detach()

    return emb, word_to_id, id_to_word


def pca_2d(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Project (V, D) -> (V, 2) with PCA via SVD.

    1. centre the data (subtract the mean of each dimension)
    2. SVD: X = U S V^T ; the rows of V^T are the principal directions
    3. project onto the top 2 directions
    """
    X = embeddings - embeddings.mean(dim=0, keepdim=True)   # (V, D) centred
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)     # Vh: (D, D)
    coords = X @ Vh[:2].T                                    # (V, 2)
    return coords


def color_for(word: str) -> str:
    if word in ROYALTY:
        return "#c0392b"   # red-ish
    if word in PETS:
        return "#2980b9"   # blue-ish
    return "#7f8c8d"       # grey for everything else


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "skipgram"
    emb, word_to_id, id_to_word = get_embeddings(variant)

    coords = pca_2d(emb).numpy()   # (V, 2)

    plt.figure(figsize=(9, 7))
    for word, idx in word_to_id.items():
        x, y = coords[idx]
        plt.scatter(x, y, c=color_for(word), s=80, edgecolors="white", linewidths=0.6)
        plt.annotate(word, (x, y), xytext=(5, 4), textcoords="offset points",
                     fontsize=11)

    plt.title(f"Word2Vec embeddings ({variant}) — PCA to 2-D", fontsize=13)
    plt.xlabel("principal component 1")
    plt.ylabel("principal component 2")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    out = f"embeddings_{variant}.png"
    plt.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
