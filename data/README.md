# Data

**Dataset:** TinyShakespeare (about 1.1M characters of public-domain Shakespeare text).

**Source:** `input.txt` from Karpathy's char-rnn repository:
<https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>

**Download:** `notebooks/01_prepare_data.ipynb` downloads the file here automatically if it is missing.

**Preprocessing:**
- Contiguous 80/10/10 train/validation/test split, so neighbouring text does not leak between sets.
- Vocabulary built from the training split only. Characters that appear only in validation or test map to an `<unk>` symbol.

Files in this folder are git-ignored, apart from this README.
