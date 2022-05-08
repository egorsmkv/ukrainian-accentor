# Accentor for Ukrainian

## Overview

This repository contains a pytorch model to make accents in Ukrainian words. 

The model was trained using this notebook: https://github.com/dsakovych/g2p_uk/blob/master/notebooks/word_stress_pytorch.ipynb

## Installation

```bash
pip install torch pandas
```

## Running

Use `accentor_cpu.py` or `accentor_gpu.py` script.

## Demo

```
python accentor_cpu.py

With stress: ['словотво́рення', 'архаї́чний', 'програ́ма', 'а-ля-фурше́т']
With pluses: ['словотв+орення', 'арха+їчний', 'прогр+ама', 'а-ля-фурш+ет']
```
