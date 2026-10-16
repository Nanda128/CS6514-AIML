# CS6514: AI Project

By Daniel Moody (23370157) & Nandakishore Vinayakrishnan (23070854)

**Research question:** with training characters held fixed, does increasing context length from 32 to 128 to 256 lower test bits-per-character for a 4-layer character transformer on TinyShakespeare?

## Running the experiment

Everything is in `pipeline.ipynb`. Open it in Colab on a T4 GPU runtime and use **Runtime → Run all** (about 1–1.5 hours). It:

1. splits TinyShakespeare into a fixed 10% test set and 10 folds of the remaining 90%;
2. trains the transformer at contexts 32, 128 and 256 on every fold (30 runs, seed f on fold f), choosing each checkpoint on its validation fold;
3. fits the n-gram reference on every fold, choosing its order and discount on validation;
4. scores everything on validation and, once, on the test set;
5. writes the analysis to `results/`:
   - `runs.csv`: one row per run, plus one per n-gram fold
   - `summary.json`: split, settings, compute, environment, statistics, error analysis and training histories
   - `figures.pdf`: all figures

Checkpoints are kept in `MyDrive/CS6514-AIML/checkpoints`. If Colab disconnects, Run all again: it resumes without retraining finished runs. Set `EVALUATE_TEST = False` in the setup cell to work on validation only. After the final run, commit the three files in `results/`.

`baseline.ipynb` is the week 5 baseline-gate record (a single context-32 run). It is not needed to reproduce the results.

## Design

| | |
| --- | --- |
| Model | GPT-style decoder, 4 layers, 4 heads, width 192, dropout 0.1, learned position embeddings, tied output weights; 1.80M (context 32) to 1.84M (context 256) parameters |
| Controlled change | Context length 32 → 128 → 256; batch size 512 / 128 / 64 so every step sees 16,384 training characters |
| Held fixed | Architecture, AdamW (lr 1e-3 → 1e-4 cosine, weight decay 0.1), gradient clip 1.0, 3,000 steps, seed per fold, evaluation targets |
| Reference conditions | Context-32 transformer; character n-gram (orders 1–8) with interpolated absolute-discounting smoothing |
| Split | Last 10% fixed test set; first 90% in 10 contiguous folds; all cuts on line breaks; vocabulary from the first 90% |
| Repeated runs | 10 per context: fold f uses seed f, so all contexts share the same seeds and folds |
| Primary metric | Fixed-target test bits-per-character: every model predicts every character from the 257th of a split onward, from the L preceding characters; BPC = cross-entropy (nats) / ln 2 |
| Model selection | Transformer: lowest sampled validation loss during training. N-gram: lowest validation BPC on its fold. The test set is never used to choose anything |

## Changes from the proposal

- **Split:** 80/10/10 became 10-fold cross-validation over the first 90%, with the last 10% as a fixed test set, so every part of the non-test text is used for validation once.
- **Repeated runs:** 5 seeds per context became 10 runs per context, one per fold with seed = fold.
- **N-gram:** tuned on each fold's validation set rather than once.

## Data and licence

TinyShakespeare from Karpathy's [char-rnn repository](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt), about 1.1M characters. The repository is MIT-licensed and the plays are public domain. The notebook downloads the file itself. No pretrained models are used.

## Compute

Colab T4 GPU. A run takes about 1.6–2 minutes, so the 30 runs take about an hour, plus a few minutes per fold for the n-gram on the CPU. Training uses fp16 mixed precision and fused scaled-dot-product attention. Exact times, hardware and library versions are recorded in `results/summary.json`.
