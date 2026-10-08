# CS6514: AI Project — context length vs. bits-per-character

By Daniel Moody (23370157) & Nandakishore Vinayakrishnan (23070854)

**Research question:** with training characters held fixed, does increasing context length from 32 to 128 to 256 lower test bits-per-character for a 4-layer character transformer on TinyShakespeare?

## Layout

| Path | Contents |
|---|---|
| `baseline.ipynb` | The whole pipeline: data, models, metric, training and validation results |
| `data/` | TinyShakespeare, downloaded by the notebook from [karpathy/char-rnn](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) (MIT licence; the plays are public domain) (git-ignored) |
| `checkpoints/` | Model weights, one folder per run (git-ignored) |
| `results/` | Data summary, n-gram grid, per-run records, validation table and plots |

## Colab

Open `baseline.ipynb`, use a T4 GPU runtime and run all cells. Clone into Google Drive so checkpoints survive a disconnect; training resumes from `last.pt`.

```python
from google.colab import drive
drive.mount("/content/drive")
!git clone <repo-url> /content/drive/MyDrive/CS6514-AIML
%cd /content/drive/MyDrive/CS6514-AIML
```
