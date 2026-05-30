"""
Step 1 — Preprocessing
======================

Goal: turn raw text into integer token IDs that a neural network can consume.

A neural net never "sees" words. It sees indices into an embedding table.
So before any training can happen, we need a two-way dictionary:

    word_to_id : "king" -> 7
    id_to_word :   7    -> "king"

This file builds that dictionary and converts our corpus into a flat list
of integer IDs.
"""

import re

# ---------------------------------------------------------------------------
# 1. A tiny corpus, kept inside the code on purpose.
#
# It is intentionally small and *thematic*: there are two loose "topics"
# (royalty and pets). Words that share a topic tend to appear near each other,
# which is exactly the co-occurrence signal Word2Vec learns from. With such a
# small corpus the embeddings won't be amazing — but the *mechanism* will be
# perfectly visible, which is the point.
# ---------------------------------------------------------------------------
TINY_CORPUS = """
the king rules the kingdom and the queen rules beside the king
the king loves the queen and the queen loves the king
a man becomes a king a woman becomes a queen
the cat chases the dog and the dog chases the cat
a cat is a small pet a dog is a loyal pet
the pet cat sleeps the pet dog sleeps near the cat
"""


def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer: lowercase, then grab runs of word characters.

    We deliberately keep this naive (no stemming, no stop-word removal) so the
    logic stays transparent. `\\w+` matches sequences of letters/digits and
    drops punctuation and whitespace.

    Shape: a single string  ->  list[str] of length N_tokens
           (N_tokens = total number of word occurrences, duplicates included)
    """
    return re.findall(r"\w+", text.lower())


def build_vocab(tokens: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """
    Build the two-way mapping between words and integer IDs.

    We assign IDs by sorting the *unique* words. Sorting isn't required for
    correctness, but it makes the IDs deterministic across runs — handy when
    you want to compare results or screenshot them.

    Let V = vocabulary size = number of UNIQUE words.

    Shapes:
        word_to_id : dict of size V   (word  -> id in [0, V-1])
        id_to_word : dict of size V   (id    -> word)
    """
    unique_words = sorted(set(tokens))
    word_to_id = {word: i for i, word in enumerate(unique_words)}
    id_to_word = {i: word for word, i in word_to_id.items()}
    return word_to_id, id_to_word


def encode(tokens: list[str], word_to_id: dict[str, int]) -> list[int]:
    """
    Replace each word with its integer ID.

    Shape: list[str] of length N_tokens  ->  list[int] of length N_tokens
           (same length — we're just relabelling, not aggregating)
    """
    return [word_to_id[token] for token in tokens]


# ---------------------------------------------------------------------------
# Quick self-check: run `python preprocess.py` to inspect the shapes directly.
# This block does not run when the file is imported elsewhere.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tokens = tokenize(TINY_CORPUS)
    word_to_id, id_to_word = build_vocab(tokens)
    token_ids = encode(tokens, word_to_id)

    V = len(word_to_id)
    print(f"N_tokens (total word occurrences) : {len(tokens)}")
    print(f"V (unique words / vocab size)     : {V}")
    print(f"first 12 tokens                   : {tokens[:12]}")
    print(f"first 12 token IDs                : {token_ids[:12]}")
    print(f"word_to_id sample                 : "
          f"{ {w: word_to_id[w] for w in list(word_to_id)[:5]} }")
