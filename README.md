# CS6514: AI Project — context length vs. bits-per-character

By Daniel Moody (23370157) & Nandakishore Vinayakrishnan (23070854)

**Research question:** with the number of training characters held fixed, does increasing context length from 32 to 128 to 256 lower test bits-per-character for a 4-layer character transformer on TinyShakespeare, by more than the variation across 5 seeds?

## Layout

| Path | Contents |
|---|---|
| `data/` | Dataset (downloaded by notebook 01); see `data/README.md` |
| `checkpoints/` | Saved model weights, one folder per run (git-ignored) |
| `results/` | Metrics, logs and plots |
| `notebooks/` | The pipeline, run in order |
| `config.yaml` | Shared experiment settings |
| `requirements.txt` | Python packages, if needed in Colab |

## Run order

1. `notebooks/01_prepare_data.ipynb`: download, split and encode TinyShakespeare
2. `notebooks/02_model.ipynb`: build the model and check initialisation
3. `notebooks/03_train_baseline.ipynb`: train every context length and seed
4. `notebooks/04_evaluate.ipynb`: compute BPC, write tables and plots

Change experiment settings in `config.yaml`, not in the notebooks.

## Colab

```python
!git clone <repo-url>
%cd <repo>/notebooks
!pip install -q -r ../requirements.txt
```
