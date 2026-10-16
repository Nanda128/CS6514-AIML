# CS6514: AI Project

By Daniel Moody (23370157) & Nandakishore Vinayakrishnan (23070854)

**Research question:** with training characters held fixed, does increasing context length from 32 to 128 to 256 lower test bits-per-character for a 4-layer character transformer on TinyShakespeare?

## Running the experiment

`baseline.ipynb` is the baseline-gate pipeline. The main comparison (contexts 32/128/256 × seeds 0, 1, 2 = 9 runs) is split across two Colab T4 notebooks so the compute is shared:

| Notebook | Runs |
| --- | --- |
| `experiment_part1.ipynb` | n-gram reference; ctx32/seed0, ctx128/seed0, ctx256/seed0, ctx256/seed1 |
| `experiment_part2.ipynb` | ctx32/seed1, ctx128/seed1, ctx32/seed2, ctx128/seed2, ctx256/seed2 |

Each notebook mounts Google Drive and keeps checkpoints in `MyDrive/CS6514-AIML`, so rerunning the training cell after a disconnect resumes from `last.pt`. When it finishes, commit `results/runs/*.json` (each includes `val_bpc`). Keep the `best.pt` checkpoints for the final test evaluation.
