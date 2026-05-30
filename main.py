"""
main.py — the whole Skip-Gram Word2Vec pipeline, end to end
===========================================================

    raw text
      -> tokenize            (preprocess.py)
      -> vocab + token IDs   (preprocess.py)
      -> (center, context)   (dataset.py)
      -> SkipGram model      (model.py)
      -> train               (train.py)
      -> word vectors        -> cosine similarity (similarity.py)

Run with:  uv run python main.py
"""

import torch

from preprocess import TINY_CORPUS, tokenize, build_vocab, encode
from dataset import make_dataloader
from train import train
from similarity import get_embeddings, most_similar

# Hyperparameters — small on purpose so it trains in seconds on CPU.
WINDOW_SIZE = 2
EMBEDDING_DIM = 16
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.01

# Fixed seed so the printed results are reproducible across runs.
torch.manual_seed(0)


def main():
    # --- Step 1: text -> token IDs -------------------------------------
    tokens = tokenize(TINY_CORPUS)
    word_to_id, id_to_word = build_vocab(tokens)
    token_ids = encode(tokens, word_to_id)
    vocab_size = len(word_to_id)
    print(f"vocab size V = {vocab_size}, corpus length N_tokens = {len(token_ids)}\n")

    # --- Step 2: (center, context) pairs -> DataLoader -----------------
    dataloader = make_dataloader(token_ids, WINDOW_SIZE, BATCH_SIZE)

    # --- Steps 3 & 4: build + train the model --------------------------
    model = train(token_ids, vocab_size, dataloader,
                  embedding_dim=EMBEDDING_DIM,
                  epochs=EPOCHS,
                  lr=LEARNING_RATE)

    # --- Step 5: inspect the learned space -----------------------------
    embeddings = get_embeddings(model)  # (V, D)
    print(f"\nlearned embedding matrix shape: {tuple(embeddings.shape)}  (V, D)\n")

    for probe in ["king", "cat", "pet"]:
        neighbours = most_similar(probe, word_to_id, id_to_word, embeddings, top_k=4)
        pretty = ", ".join(f"{w} ({s:.2f})" for w, s in neighbours)
        print(f"most similar to '{probe}': {pretty}")


if __name__ == "__main__":
    main()
