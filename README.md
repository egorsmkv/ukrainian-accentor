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

With stress: Я́ співа́ю ве́селу пі́сню в Украї́ні
With pluses: +Я спів+аю в+еселу п+існю в Укра+їні
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

text = "Я співаю веселу пісню в Україні"

stressed_words = accentor.process(text, mode='stress')
plused_words = accentor.process(text, mode='plus')

print('With stress:', stressed_words)
print('With pluses:', plused_words)
```

## Attribution
Sentence support implemented by code from [tokenize-uk](https://github.com/lang-uk/tokenize-uk) by [lang-uk](https://github.com/lang-uk)