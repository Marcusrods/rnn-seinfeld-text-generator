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

## Training run

Trained on `data/Seinfeld_short.txt` (313k tokens, ~28k unique words) on Apple Silicon (MPS backend), 5 epochs, ~10 minutes total. Final training loss 4.07, down from 5.58 at the end of epoch 1. Random baseline for a 28k-word vocabulary is `ln(28000) ~= 10.2`, so the model has learned real local patterns. Below baseline-quality for coherent dialogue, on the boundary of "decent" for a portfolio-scale run on a small corpus.

## Sample output

Three samples committed under `docs/` from a single trained checkpoint, prime word `jerry`, generation length 400 tokens, varying temperature:

- [`docs/sample_output_temp0.7.txt`](docs/sample_output_temp0.7.txt) - conservative (temp 0.7), repetitive but locally coherent.
- [`docs/sample_output.txt`](docs/sample_output.txt) - default (temp 1.0), the headline sample.
- [`docs/sample_output_temp1.2.txt`](docs/sample_output_temp1.2.txt) - exploratory (temp 1.2), more variety, more nonsense.

First few lines at default temperature 1.0:

```
jerry: you want me to be a little thing?

kramer: well, i don't know, i don't want to be a good.

george: i can't have to have it?

elaine: (to jerry) hey, you know, you know you have. i got it.

jerry: well, it's the guy..

elaine: i know, i'm going to be a little thing?(jerry enters.

jerry: i was not a lot.

george: well.

jerry: (looking.)

jerry: i don't understand, you have to have it.
```

Reads like Seinfeld parody written by someone with mild concussion. Character names are correct, punctuation tokens render back to symbols, dialogue rhythm survives. Plot coherence does not. Exactly what an LSTM language model trained for 5 epochs on a small corpus produces, and a useful baseline for the "what would a transformer give us instead" question in the extension list below.

## What I'd extend

- Switch from LSTM to a small transformer decoder. The character-by-character Seinfeld corpus is too small for a serious transformer, but on the long-text version a 4-layer GPT-style model would noticeably outperform the LSTM on perplexity.
- Add validation perplexity tracking, not just training loss. The current loop trains for one epoch with no held-out check.
- Persist the vocabulary separately from the model state, so a finer-grained sampling UI doesn't need to reload the full pickle.
- Move to subword tokenisation (BPE) so the model handles OOV words at inference time, instead of crashing on prime words that didn't appear in training.
