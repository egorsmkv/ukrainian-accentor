# Accentor for Ukrainian

## Overview

This repository contains a PyTorch model to make accents in Ukrainian words.

The model was trained using this
notebook: https://github.com/dsakovych/g2p_uk/blob/master/notebooks/word_stress_pytorch.ipynb

## Installation

```bash
pip install -r requirements.txt
```

## Demo

```
python inference.py

With stress: ['словотво́рення', 'архаї́чний', 'програ́ма', 'а-ля-фурше́т']
With pluses: ['словотв+орення', 'арха+їчний', 'прогр+ама', 'а-ля-фурш+ет']
```

Or you can use the library like the following:

```python
import torch

importer = torch.package.PackageImporter("accentor-lite.pt")
accentor = importer.load_pickle("uk-accentor", "model")

# Using GPU
# accentor.cuda()
# Back to CPU
# accentor.cpu()

test_words1 = ["словотворення", "архаїчний", "програма", "а-ля-фуршет"]

stressed_words = accentor.predict(test_words1, mode='stress')
plused_words = accentor.predict(test_words1, mode='plus')

print('With stress:', stressed_words)
print('With pluses:', plused_words)
```
