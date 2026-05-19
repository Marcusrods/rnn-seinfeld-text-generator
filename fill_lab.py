"""Build the filled Seinfeld RNN/LSTM lab notebook from the lecturer template.

Reads ``rnn_seinfeld_lab_template.ipynb``, replaces the four TODO code cells
(`create_lookup_tables`, `token_lookup`, `batch_data`, `RNN`, and
`optimization_step`) with my implementations, writes
``rnn_seinfeld_lab.ipynb``.

Run once: ``python fill_lab.py``.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


CREATE_LOOKUP_TABLES = '''def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary.

    :param text: The text of tv scripts split into words (list of str).
    :return: A tuple of dicts (vocab_to_int, int_to_vocab).
    """
    # Stable sort by frequency descending, with alphabetical tie-break, so the
    # lookup tables are deterministic across runs.
    from collections import Counter

    counts = Counter(text)
    sorted_words = sorted(counts.keys(), key=lambda w: (-counts[w], w))
    vocab_to_int = {word: idx for idx, word in enumerate(sorted_words)}
    int_to_vocab = {idx: word for word, idx in vocab_to_int.items()}
    return vocab_to_int, int_to_vocab
'''


TOKEN_LOOKUP = '''def token_lookup():
    """
    Generate a dict mapping punctuation to tokens.

    :return: Dict where the key is the punctuation and the value is the token.
    """
    return {
        ".": "||period||",
        ",": "||comma||",
        '"': "||quotation_mark||",
        ";": "||semicolon||",
        "!": "||exclamation_mark||",
        "?": "||question_mark||",
        "(": "||left_parentheses||",
        ")": "||right_parentheses||",
        "-": "||dash||",
        "\\n": "||return||",
    }
'''


BATCH_DATA = '''from torch.utils.data import TensorDataset, DataLoader


def batch_data(words, sequence_length: int, batch_size: int) -> DataLoader:
    """Build a DataLoader of (inputs, targets) pairs for causal LM training.

    Each sample is a stride-1 sliding window over ``words``:

        inputs[i]  = words[i     : i + T]
        targets[i] = words[i + 1 : i + 1 + T]

    where T = sequence_length. The dataset contains exactly
    ``len(words) - T`` windows.
    """
    words_tensor = torch.as_tensor(words, dtype=torch.long)
    n_windows = words_tensor.size(0) - sequence_length
    if n_windows <= 0:
        raise ValueError(
            f"Not enough tokens ({words_tensor.size(0)}) for sequence_length="
            f"{sequence_length}."
        )

    # Materialise inputs/targets as (n_windows, T) tensors.
    inputs = torch.stack([
        words_tensor[i : i + sequence_length] for i in range(n_windows)
    ])
    targets = torch.stack([
        words_tensor[i + 1 : i + 1 + sequence_length] for i in range(n_windows)
    ])

    dataset = TensorDataset(inputs, targets)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
'''


RNN_CLASS = '''import torch.nn as nn
from torch import Tensor
from typing import Tuple, Union

# RNN/GRU hidden: Tensor. LSTM hidden: tuple(h_0, c_0).
RNNHidden = Union[Tensor, Tuple[Tensor, Tensor]]


class RNN(nn.Module):
    """Word-level next-token language model with an LSTM backbone."""

    def __init__(
        self,
        vocab_size: int,
        output_size: int,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, nn_input: Tensor, hidden: RNNHidden) -> Tuple[Tensor, RNNHidden]:
        """Compute next-token logits at every timestep.

        Shape flow: (B, T) -> (B, T, E) -> (B, T, H) -> (B, T, V)
        """
        embeds = self.embedding(nn_input)              # (B, T, E)
        out, hidden = self.lstm(embeds, hidden)        # (B, T, H)
        out = self.dropout(out)
        logits = self.fc(out)                          # (B, T, V)
        return logits, hidden

    def init_hidden(self, batch_size: int) -> RNNHidden:
        """Initial recurrent state (zeros on the parameter device/dtype)."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, dtype=dtype, device=device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, dtype=dtype, device=device)
        return (h0, c0)
'''


OPTIMIZATION_STEP = '''def optimization_step(rnn, optimizer, criterion, inp, target, hidden):
    """Run one optimization step of the causal language model.

    Returns ``(loss_value, new_hidden)`` where ``loss_value`` is averaged over
    ``batch * seq_len`` next-token predictions.
    """
    rnn.train()
    inp = inp.to(next(rnn.parameters()).device)
    target = target.to(next(rnn.parameters()).device)

    # Detach hidden state from the previous batch's graph.
    if isinstance(hidden, tuple):
        hidden = tuple(h.detach() for h in hidden)
    else:
        hidden = hidden.detach()

    optimizer.zero_grad()
    logits, hidden = rnn(inp, hidden)                  # (B, T, V)

    # Flatten for CE: logits (B*T, V), target (B*T,)
    B, T, V = logits.shape
    loss = criterion(logits.reshape(B * T, V), target.reshape(B * T))

    loss.backward()
    nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=5.0)
    optimizer.step()

    return loss.item(), hidden
'''


REPLACEMENTS = {
    "def create_lookup_tables(text):": CREATE_LOOKUP_TABLES,
    "def token_lookup():": TOKEN_LOOKUP,
    "def batch_data(words, sequence_length: int, batch_size: int) -> DataLoader:": BATCH_DATA,
    "class RNN(nn.Module):": RNN_CLASS,
    "def optimization_step(rnn, optimizer, criterion, inp, target, hidden):": OPTIMIZATION_STEP,
}


def replace_cell_source(cell: dict, new_source: str) -> None:
    """Replace ``cell['source']`` while preserving notebook list-of-strings shape."""
    cell["source"] = new_source.splitlines(keepends=True)
    cell.setdefault("outputs", [])
    cell["outputs"] = []
    cell["execution_count"] = None


def fill_notebook(template_path: Path, out_path: Path) -> None:
    nb = json.loads(template_path.read_text(encoding="utf-8"))
    nb = copy.deepcopy(nb)

    replaced = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        for marker, replacement in REPLACEMENTS.items():
            if marker in src and "TODO" in src:
                replace_cell_source(cell, replacement)
                replaced += 1
                break

    out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Replaced {replaced} TODO cells. Wrote {out_path}.")


if __name__ == "__main__":
    here = Path(__file__).parent
    fill_notebook(
        here / "rnn_seinfeld_lab_template.ipynb",
        here / "rnn_seinfeld_lab.ipynb",
    )
