"""
main_cbow.py — CBOW Word2Vec, end to end
========================================

CBOW = Continuous Bag Of Words: average the surrounding words to predict the
centre word (the mirror image of Skip-Gram).

Reuses preprocess.py and similarity.py unchanged; swaps in the CBOW dataset,
model and trainer.

Run with:  uv run python main_cbow.py
"""

import torch

from preprocess import TINY_CORPUS, tokenize, build_vocab, encode
from dataset_cbow import make_cbow_dataloader
from train_cbow import train_cbow
from similarity import most_similar

WINDOW_SIZE = 2
EMBEDDING_DIM = 16
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.01

torch.manual_seed(0)


def main():
    tokens = tokenize(TINY_CORPUS)
    word_to_id, id_to_word = build_vocab(tokens)
    token_ids = encode(tokens, word_to_id)
    vocab_size = len(word_to_id)
    print(f"vocab size V = {vocab_size}, corpus length N_tokens = {len(token_ids)}\n")

    dataloader = make_cbow_dataloader(token_ids, WINDOW_SIZE, BATCH_SIZE)

    model = train_cbow(vocab_size, dataloader,
                       embedding_dim=EMBEDDING_DIM,
                       epochs=EPOCHS,
                       lr=LEARNING_RATE)

    embeddings = model.embeddings.weight.detach()  # (V, D)
    print(f"\nlearned embedding matrix shape: {tuple(embeddings.shape)}  (V, D)\n")

    for probe in ["king", "cat", "pet"]:
        neighbours = most_similar(probe, word_to_id, id_to_word, embeddings, top_k=4)
        pretty = ", ".join(f"{w} ({s:.2f})" for w, s in neighbours)
        print(f"most similar to '{probe}': {pretty}")


if __name__ == "__main__":
    main()
