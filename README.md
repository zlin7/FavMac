# Fast Online Value-Maximizing Prediction Sets with Conformal Cost Control (FavMac)
ICML 2023 [arxiv](https://arxiv.org/abs/2302.00839).


# Dependency:

Install
`pip install persist-to-disk` [link](https://pypi.org/project/persist-to-disk/) for caching purposes.
The full environment is in `env.yml`.

# Settings

For logging purposes, register on neptune.ai and supply the `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN` in `_settings.py`.

# Quick Demo

Use `notebook/demo.ipynb` to see how FavMac is employed.
You only need to supply the prediction and labels.

# Replicate our experiments

*Note that "value" in our paper corresponds to "util" (or utility) in our code*.
Apology for any confusion.

### Process MIMIC-III
For the MIMIC-III data, download and put it in `MIMIC_PATH`  in `_settings.py` (no need to unzip each table).
Set `MIMIC_PREPROCESS_OUTPUT` to where you want to store the processed data (from below).
Then, do the following to preprocess all MIMIC data.
```
python -m data_utils._cache_all
```
Also, download the "Clinical BERT Weights" from https://github.com/kexinhuang12345/clinicalBERT to `PRETRAINED_PATH` in `models/clinicalBERT`.

### Process MNIST
run `data_utils/preprocessing/mnist_multilabel.py` to generate the superimposed MNIST images.

### Train the models

```
python -m pipeline.train_mnist
python -m pipeline.train_mimic3hcc
```
Once you finished, each config function will generate a "key" (something like `EHRModel-MIMIC-IIICompletion-20230521_160512091696`).
Supply this to `scripts.evaluated.Config.KEYS`.

To train the deepset model used in the FPCP baseline, use `scripts/train_deepsets.py`.
You will also need to update the `dsproxy_*` keys in `scripts.evaluated.Config.KEYS`.

### Cost-control experiments

After training is done, copy paste the keys from the previous steps into `scripts.evaluate.Config.KEYS`.
Then, you could cache the experiment results like the examples in `__main__` of `scripts.evaluate`.
Alternatively, you can directly use `notebook/MIMIC.ipynb` or `notebook/MNIST.ipynb` to run the experiment and see the results.
For simplicity I set the number of seets to 3 instead of 10 in `scripts.evaluate`.
