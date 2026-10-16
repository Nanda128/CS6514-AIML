# CS6514: AI Project

By Daniel Moody (23370157) & Nandakishore Vinayakrishnan (23070854)

**Research question:** with training characters held fixed, does increasing context length from 32 to 128 to 256 lower test bits-per-character for a 4-layer character transformer on TinyShakespeare?

## Running the experiment

Everything is in `pipeline.ipynb`. Open it in Colab on a T4 GPU runtime and use **Runtime → Run all** (about 1–1.5 hours). It:

1. splits TinyShakespeare into a fixed 10% test set and 10 folds of the remaining 90%;
2. trains the transformer at contexts 32, 128 and 256 on every fold (30 runs, seed f on fold f), choosing each checkpoint on its validation fold;
3. fits the n-gram reference on every fold;
4. scores everything on validation and, once, on the test set;
5. writes the analysis to `results/`:
   - `runs.csv`: one row per run, plus one per n-gram fold
   - `summary.json`: split, settings, statistics, error analysis and training histories
   - `figures.pdf`: all figures

Checkpoints are kept in `MyDrive/CS6514-AIML/checkpoints`; if Colab disconnects, Run all again and it resumes without retraining finished runs.
