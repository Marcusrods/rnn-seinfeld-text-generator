# rnn-seinfeld-text-generator

A word-level RNN language model that generates Seinfeld TV scripts. Built end to end: text preprocessing, vocabulary construction, causal LM batching, LSTM in PyTorch, training with gradient clipping, temperature-controlled sampling.

Submitted as the BTS-AI sequence lab on the Masters in Big Data and AI Artificial Intelligence module.

## What this proves

- **Sequence modelling in PyTorch.** Embedding layer, multi-layer LSTM, dropout regularisation, linear projection head.
- **Causal language modelling done correctly.** Targets are inputs shifted by one position. Loss is computed on every timestep (`B * T` predictions per batch). Hidden state is detached between batches.
- **Training hygiene.** Cross-entropy loss with logits flattening, gradient clipping at norm 5, Adam optimiser, device-aware (CUDA, Apple MPS, or CPU).
- **Text preprocessing.** Punctuation tokenisation so the model learns punctuation as its own tokens, deterministic vocabulary construction sorted by frequency with alphabetical tie-break.
- **Sampling.** Temperature-controlled top-k sampling for generation. The script prime word seeds the first prediction; subsequent predictions roll forward on the sliding window.

## Attribution

The lab notebook scaffold, the dataset files (`Seinfeld_short.txt`, `Seinfeld_long.txt`), and the utility code (preprocessing helpers, `train_loop`, `generate`) are from my lecturer **Ricardo Garcia Gutierrez** for the BTS Masters AI module.

**My contribution** is the five filled-in components:

1. `create_lookup_tables` - frequency-sorted vocabulary with deterministic tie-break.
2. `token_lookup` - punctuation-to-token map covering ten symbols.
3. `batch_data` - causal LM DataLoader with stride-1 sliding windows.
4. `RNN` class - LSTM language model.
5. `optimization_step` - one training step with detached hidden state, gradient clipping, batch-time flatten for cross-entropy.

These are isolated into `fill_lab.py` so the lecturer's template stays as the source of truth in `rnn_seinfeld_lab_template.ipynb`, and the filled notebook `rnn_seinfeld_lab.ipynb` is regenerable.

## How to run

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name rnn-seinfeld --display-name "Python (rnn-seinfeld)"

# Open the filled notebook and run top to bottom
jupyter notebook rnn_seinfeld_lab.ipynb
```

The notebook saves the trained model to `save/trained_rnn.pt` and then runs the `generate` cell to produce sample dialogue.

To regenerate the filled notebook after changing implementations:

```bash
python fill_lab.py
```

## Hyperparameters

The shipped defaults are sized to overfit on the short script in a few minutes on Apple Silicon:

```
sequence_length = 10
batch_size = 128
embedding_dim = 128
hidden_dim = 128
n_layers = 2
dropout = 0.5
learning_rate = 0.001
num_epochs = 1
```

For real generation quality, switch to `Seinfeld_long.txt`, raise `num_epochs` to 5 or 10, increase `hidden_dim` to 256 or 512.

## Sample output

Once trained, the `generate` cell produces continuations that read like Seinfeld stage direction more than dialogue, but the names, beats, and rhythm survive the round trip through the LSTM. Sample to be added once the full long-text training run completes.

## What I'd extend

- Switch from LSTM to a small transformer decoder. The character-by-character Seinfeld corpus is too small for a serious transformer, but on the long-text version a 4-layer GPT-style model would noticeably outperform the LSTM on perplexity.
- Add validation perplexity tracking, not just training loss. The current loop trains for one epoch with no held-out check.
- Persist the vocabulary separately from the model state, so a finer-grained sampling UI doesn't need to reload the full pickle.
- Move to subword tokenisation (BPE) so the model handles OOV words at inference time, instead of crashing on prime words that didn't appear in training.
