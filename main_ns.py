"""
main_ns.py — Skip-Gram with Negative Sampling, end to end
=========================================================

Reuses preprocess.py, dataset.py and similarity.py unchanged. The only swapped
parts are the model (model_ns.py) and the trainer (train_ns.py).

The learned word vectors are the INPUT embedding table (out_embeddings is
discarded by convention).

Run with:  uv run python main_ns.py
"""

import torch

from preprocess import TINY_CORPUS, tokenize, build_vocab, encode
from dataset import make_dataloader
from train_ns import train_ns
from similarity import most_similar

# Hyperparameters.
WINDOW_SIZE = 2
EMBEDDING_DIM = 16
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.01
K_NEGATIVES = 5          # negatives drawn per positive pair

torch.manual_seed(0)


def main():
    # text -> token IDs
    tokens = tokenize(TINY_CORPUS)
    word_to_id, id_to_word = build_vocab(tokens)
    token_ids = encode(tokens, word_to_id)
    vocab_size = len(word_to_id)
    print(f"vocab size V = {vocab_size}, corpus length N_tokens = {len(token_ids)}")
    print(f"negatives per positive K = {K_NEGATIVES}\n")

    # (center, context) pairs -> DataLoader  (same as the softmax version)
    dataloader = make_dataloader(token_ids, WINDOW_SIZE, BATCH_SIZE)

    # build + train the SGNS model
    model = train_ns(token_ids, vocab_size, dataloader,
                     embedding_dim=EMBEDDING_DIM,
                     epochs=EPOCHS,
                     lr=LEARNING_RATE,
                     k_negatives=K_NEGATIVES)

    # the kept word vectors = input embedding table
    embeddings = model.in_embeddings.weight.detach()  # (V, D)
    print(f"\nlearned embedding matrix shape: {tuple(embeddings.shape)}  (V, D)\n")

    for probe in ["king", "cat", "pet"]:
        neighbours = most_similar(probe, word_to_id, id_to_word, embeddings, top_k=4)
        pretty = ", ".join(f"{w} ({s:.2f})" for w, s in neighbours)
        print(f"most similar to '{probe}': {pretty}")


if __name__ == "__main__":
    main()
