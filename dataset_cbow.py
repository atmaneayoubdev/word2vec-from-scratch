"""
CBOW dataset — (context words -> center word)
=============================================

CBOW is Skip-Gram's mirror image. Skip-Gram turned one center into many
(center, context) pairs. CBOW does the opposite: it collects ALL the context
words around a position into one bag, and the label is the single center word.

    ... the king [rules] the kingdom ...   (window_size = 2)
    context = [the, king, the, kingdom]  ->  center = rules

So each example has 2 * window_size input words and one target.

Simplification for clean batching: we only emit examples where the full window
fits (centre positions window_size .. N-window_size-1). That way every example
has exactly C = 2*window_size context words, so a batch is a neat (batch, C)
tensor with no padding. Edge words are skipped — fine for a teaching corpus.
"""

import torch
from torch.utils.data import Dataset, DataLoader


def make_cbow_examples(token_ids: list[int], window_size: int):
    """
    Build (context_ids, center_id) examples.

    Shapes:
        each context_ids -> list[int] of length C = 2*window_size
        each center_id   -> int
    """
    C = 2 * window_size
    examples = []
    # only positions with a full window on both sides
    for center_pos in range(window_size, len(token_ids) - window_size):
        context = []
        for offset in range(-window_size, window_size + 1):
            if offset == 0:
                continue  # skip the center itself
            context.append(token_ids[center_pos + offset])
        assert len(context) == C
        examples.append((context, token_ids[center_pos]))
    return examples


class CBOWDataset(Dataset):
    """
    Each item:
        context : LongTensor (C,)   the surrounding word IDs (the input bag)
        center  : LongTensor ()     the word to predict (the label)
    """

    def __init__(self, examples):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        context, center = self.examples[idx]
        return torch.tensor(context, dtype=torch.long), \
               torch.tensor(center, dtype=torch.long)


def make_cbow_dataloader(token_ids, window_size: int, batch_size: int) -> DataLoader:
    """
    After collation a batch is:
        contexts : (batch, C)   each row is one bag of context word IDs
        centers  : (batch,)     the target center word per row
    """
    examples = make_cbow_examples(token_ids, window_size)
    dataset = CBOWDataset(examples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------
# Self-check.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from preprocess import TINY_CORPUS, tokenize, build_vocab, encode

    tokens = tokenize(TINY_CORPUS)
    word_to_id, id_to_word = build_vocab(tokens)
    token_ids = encode(tokens, word_to_id)

    WINDOW_SIZE = 2
    examples = make_cbow_examples(token_ids, WINDOW_SIZE)
    print(f"C (context words per example) : {2 * WINDOW_SIZE}")
    print(f"number of examples            : {len(examples)}")

    print("\nfirst 3 examples (as words):")
    for context, center in examples[:3]:
        ctx_words = [id_to_word[i] for i in context]
        print(f"  context={ctx_words}  ->  center={id_to_word[center]}")

    loader = make_cbow_dataloader(token_ids, WINDOW_SIZE, batch_size=8)
    contexts, centers = next(iter(loader))
    print(f"\none batch -> contexts shape {tuple(contexts.shape)}, "
          f"centers shape {tuple(centers.shape)}")
