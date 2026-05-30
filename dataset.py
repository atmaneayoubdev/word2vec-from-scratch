"""
Step 2 — Skip-Gram pairs + PyTorch Dataset / DataLoader
=======================================================

Skip-Gram's task: given a CENTER word, predict a nearby CONTEXT word.

We slide a window of radius `window_size` over the token-ID stream. For every
center position we emit one (center, context) pair per neighbour inside the
window. One sentence therefore explodes into many tiny supervised examples.

Both elements of a pair are just integer IDs:
    (center_id, context_id)  e.g.  (16, 9)  meaning  ("rules", "kingdom")
"""

import torch
from torch.utils.data import Dataset, DataLoader


def make_skipgram_pairs(token_ids: list[int], window_size: int) -> list[tuple[int, int]]:
    """
    Build (center, context) pairs by sliding a window over the token stream.

    For a center at position i, the context positions are
        i-window_size, ..., i-1, i+1, ..., i+window_size
    (we skip i itself — a word is not its own context).

    Near the start/end of the stream the window is naturally truncated by the
    range bounds, so edge words simply produce fewer pairs.

    Shape: list[int] of length N_tokens  ->  list of P pairs
           P is roughly N_tokens * (2 * window_size), minus edge truncation
           and minus the center itself.
    """
    pairs = []
    for center_pos in range(len(token_ids)):
        center_id = token_ids[center_pos]

        start = max(0, center_pos - window_size)
        end = min(len(token_ids), center_pos + window_size + 1)

        for context_pos in range(start, end):
            if context_pos == center_pos:
                continue  # a word is not its own context
            context_id = token_ids[context_pos]
            pairs.append((center_id, context_id))

    return pairs


class SkipGramDataset(Dataset):
    """
    Wrap the list of (center, context) pairs in a torch Dataset.

    A Dataset only needs to answer two questions:
        __len__  : how many examples are there?
        __getitem__(idx) : give me example #idx as tensors.

    Each example is two scalar LongTensors (integer IDs). They're Long because
    they will be used as *indices* into the embedding table — and indexing
    requires integer types, not floats.
    """

    def __init__(self, pairs: list[tuple[int, int]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        center_id, context_id = self.pairs[idx]
        # scalar tensors, shape ()  — dtype long for use as indices / class labels
        return torch.tensor(center_id, dtype=torch.long), \
               torch.tensor(context_id, dtype=torch.long)


def make_dataloader(token_ids: list[int], window_size: int, batch_size: int) -> DataLoader:
    """
    Convenience: pairs -> Dataset -> DataLoader.

    The DataLoader batches the scalar examples together and shuffles them.
    After collation a batch is:
        centers  : LongTensor of shape (batch_size,)
        contexts : LongTensor of shape (batch_size,)

    Shuffling matters: consecutive pairs come from the same sentence, so
    without shuffling each batch would be highly correlated and gradients
    would be biased toward whatever local region we're sweeping through.
    """
    pairs = make_skipgram_pairs(token_ids, window_size)
    dataset = SkipGramDataset(pairs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------
# Self-check: inspect the pairs and one collated batch.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from preprocess import TINY_CORPUS, tokenize, build_vocab, encode

    tokens = tokenize(TINY_CORPUS)
    word_to_id, id_to_word = build_vocab(tokens)
    token_ids = encode(tokens, word_to_id)

    WINDOW_SIZE = 2
    pairs = make_skipgram_pairs(token_ids, WINDOW_SIZE)

    print(f"N_tokens : {len(token_ids)}")
    print(f"P pairs  : {len(pairs)}  (window_size={WINDOW_SIZE})")

    # Show the first few pairs as words so the task is human-readable.
    print("\nfirst 6 pairs (as words):")
    for c, ctx in pairs[:6]:
        print(f"  center={id_to_word[c]:<8} -> context={id_to_word[ctx]}")

    loader = make_dataloader(token_ids, WINDOW_SIZE, batch_size=8)
    centers, contexts = next(iter(loader))
    print(f"\none batch -> centers shape {tuple(centers.shape)}, "
          f"contexts shape {tuple(contexts.shape)}")
    print(f"centers  : {centers.tolist()}")
    print(f"contexts : {contexts.tolist()}")
