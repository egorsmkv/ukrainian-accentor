# Accentor for Ukrainian

## Overview

This repository contains a PyTorch model to make accents in Ukrainian words.

The model was trained using this
notebook: https://github.com/dsakovych/g2p_uk/blob/master/notebooks/word_stress_pytorch.ipynb

## Installation

```bash
pip install torch pandas
```

## Demo

```
python accentor.py

With stress: ['словотво́рення', 'архаї́чний', 'програ́ма', 'а-ля-фурше́т']
With pluses: ['словотв+орення', 'арха+їчний', 'прогр+ама', 'а-ля-фурш+ет']
```

Or you can use the library like the following:

```python
from accentor import Accentor, replace_accents

accentor = Accentor('./accentor.pt', './dict.txt')

test_words1 = ["словотворення", "архаїчний", "програма", "а-ля-фуршет"]

stressed_words = accentor.predict(test_words1, mode='stress')
plused_words = [replace_accents(x) for x in stressed_words]

print('With stress:', stressed_words)
print('With pluses:', plused_words)
```
