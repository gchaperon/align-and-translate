[![Code Quality](https://github.com/gchaperon/align-and-translate/actions/workflows/lint.yaml/badge.svg)](https://github.com/gchaperon/align-and-translate/actions/workflows/lint.yaml?query=branch%3Amaster)
[![Static Types](https://github.com/gchaperon/align-and-translate/actions/workflows/types.yaml/badge.svg)](https://github.com/gchaperon/align-and-translate/actions/workflows/types.yaml?query=branch%3Amaster)
# align-and-translate
My replication code for the paper [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473).


## Install
If you want to retrain the model, and replicate the results, install with
```console
pip install '.[train]'
```

## The data
Download all data with
```console
dvc pull
```
This downloads the preprocessed dataset and a tokenizer trained on the data.
The data uses around 14Gb of disk space.


## Changes
This section sumarizes the difference between my implementation and the one
described in the paper.

* Train dataset size reduction, from 850M words to 348M words. I didn't do this
  step because I considered it too difficult, and also because the objective of
  this repo is to replicate the architecture.
