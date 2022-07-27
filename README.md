# Accentor for Ukrainian

## Overview

This repository contains a PyTorch model to make accents in Ukrainian words.

The model was trained using [this notebook](https://github.com/egorsmkv/ukrainian-accentor/blob/main/notebooks/word_stress_pytorch.ipynb) in `notebooks/` folder

## Alternatives

- [Ukrainian word stress](https://github.com/lang-uk/ukrainian-word-stress) by Oleksiy Syvokon

## Installation

```bash
pip install git+https://github.com/egorsmkv/ukrainian-accentor.git
```

## Demo

```
python inference.py

With stress: Я́ співа́ю ве́селу пі́сню в Украї́ні
With pluses: +Я спів+аю в+еселу п+існю в Укра+їні
```

Or you can use the library like the following:

```python
import ukrainian_accentor as accentor

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

## Development

If you want do train your own model clone this repo and run:

```bash
pip install -r requirements.dev.txt
```

## Attribution
Training script is a derivative of [word_stress_pytorch.ipynb](https://github.com/dsakovych/g2p_uk/blob/master/notebooks/word_stress_pytorch.ipynb) by [dsakovych](https://github.com/dsakovych)

Sentence support implemented by code from [tokenize-uk](https://github.com/lang-uk/tokenize-uk) by [lang-uk](https://github.com/lang-uk)
