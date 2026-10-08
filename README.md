# CS6514: AI Project — context length vs. bits-per-character

By Daniel Moody (23370157) & Nandakishore Vinayakrishnan (23070854)

**Research question:** with training characters held fixed, does increasing context length from 32 to 128 to 256 lower test bits-per-character for a 4-layer character transformer on TinyShakespeare?

## Layout

| Path | Contents |
|---|---|
| `data/` | TinyShakespeare, downloaded by notebook 01 from [karpathy/char-rnn](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) (MIT licence; the plays are public domain) (git-ignored) |
| `checkpoints/` | Model weights, one folder per run (git-ignored) |
| `results/` | Data summary, n-gram grid, per-run records, validation table and plots |
| `notebooks/` | The pipeline, run in order |

## Run order

1. `01_prepare_data.ipynb`: download, then contiguous 80/10/10 split on line boundaries, vocabulary from the training split
2. `02_model.ipynb`: experiment settings, transformer, n-gram and fixed-target BPC metric (loaded by 03 and 04)
3. `03_train_baseline.ipynb`: tune the n-gram on validation, train the context-32 transformer
4. `04_evaluate.ipynb`: validation BPC table and training curves

## Colab

Use a T4 GPU runtime. Clone into Google Drive so checkpoints survive a disconnect; training resumes from `last.pt`.

```python
from google.colab import drive
drive.mount("/content/drive")
!git clone <repo-url> /content/drive/MyDrive/CS6514-AIML
%cd /content/drive/MyDrive/CS6514-AIML/notebooks
```
