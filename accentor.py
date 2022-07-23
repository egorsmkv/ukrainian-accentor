from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from word_tokenizer import tokenize, detokenize

EXCLUDE_LIST = [
    '\t', '$', '%', '-', '\\', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
    'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
    'v', 'w', 'x', 'y', 'z', '~', '\xa0', '\xad', '·', 'ђ', 'ј', 'љ', 'њ', 'ћ',
    'ў', 'џ', '–', '‘', '’', '“', '”', '•', '№',
    'ъ', 'ы', 'э'
]


class SequenceTokenizer:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.oov_token = '<UNK>'
        self.oov_token_index = 0

    def fit(self, sequence):
        self.index2word = dict(enumerate([self.oov_token] + sorted(set(self.flatten(sequence))), 1))
        self.word2index = {v: k for k, v in self.index2word.items()}
        self.oov_token_index = self.word2index.get(self.oov_token)
        return self

    def transform(self, X):
        res = []
        for line in X:
            res.append([self.word2index.get(item, self.oov_token_index) for item in line])
        return res

    def flatten(self, arr):
        for item in arr:
            if isinstance(item, list):
                yield from self.flatten(item)
            else:
                yield item


def remove_accents(x):
    chars = list(x.encode('utf-8').replace(b'\xcc\x81', b'').decode('utf-8'))

    final_chars = []
    for i, c in enumerate(chars):
        if c == '+':
            try:
                tmp = final_chars[i - 1]

                final_chars.pop()
                final_chars.append('+')
                final_chars.append(tmp)
            except IndexError:
                final_chars.append(c)
        else:
            final_chars.append(c)

    return ''.join(final_chars)


def replace_accents(x):
    chars = list(x.encode('utf-8').replace(b'\xcc\x81', b'+').decode('utf-8'))

    final_chars = []
    for i, c in enumerate(chars):
        if c == '+':
            try:
                tmp = final_chars[i - 1]

                final_chars.pop()
                final_chars.append('+')
                final_chars.append(tmp)
            except IndexError:
                final_chars.append(c)
        else:
            final_chars.append(c)

    return ''.join(final_chars)


def stress_pos(x):
    try:
        res = np.zeros(len(x) - 1)
        stress_idx = x.find('+')
        res[stress_idx] = 1
        return res
    except IndexError:
        return np.NaN


class LSTM_model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(LSTM_model, self).__init__()
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(input_size=self.embeddings.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=3,
                            batch_first=True,
                            bidirectional=True,
                            dropout = 0.05)
        self.linear = nn.Linear(self.hidden_dim * 8 , 64)
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim * 8, affine=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(64, target_size)

    def forward(self, x):
        h_embeddings = self.embeddings(x)

        h_lstm, _ = self.lstm(h_embeddings)
        d_1 = h_lstm[:,0,:]
        d_2 = h_lstm[:,h_lstm.shape[1]//4,:]
        d_3 = h_lstm[:,h_lstm.shape[1]*3//4,:]
        d_4 = h_lstm[:,-1,:]

        x = torch.cat((d_1, d_2, d_3, d_4), 1)
        x = self.batch_norm(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        y = torch.softmax(x, dim=1)

        return y


class Accentor:
    dict: List[Tuple[str, str]]
    tokenizer: SequenceTokenizer
    max_sequence_len: int
    model: LSTM_model
    use_cuda: bool

    def __init__(self, model_path: str, dict_path: str, use_cuda: bool = False):
        dictionary = self.load_dict(dict_path)

        data = pd.DataFrame(dictionary)
        data.columns = ['word', 'word_with_stress']

        for item in EXCLUDE_LIST:
            data = data.loc[~data.word.str.contains(item, regex=False)]

        data['word_list'] = data.word.map(list)
        data['word_stress_pos'] = data.word_with_stress.map(stress_pos)

        data = data.loc[data.word_stress_pos.notna()]

        self.max_sequence_len = np.max(data.word.str.len())

        self.tokenizer = SequenceTokenizer()
        self.tokenizer.fit(data.word_list)

        self.model = LSTM_model(embedding_dim=64,
                                hidden_dim=64,
                                vocab_size=len(self.tokenizer.word2index) + 1,
                                target_size=self.max_sequence_len)
        if use_cuda:
            self.cuda()
        else:
            self.cpu()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def cuda(self):
        self._cuda = True
        self.model.cuda()

    def cpu(self):
        self._cuda = False
        self.model.cpu()

    def pad_sequence(self, lst):
        if isinstance(lst[0], list):
            return np.array([i[:self.max_sequence_len] + 
                            [0] * (self.max_sequence_len - len(i)) for i in lst])

        raise ValueError('lst is incorrect')

    def predict(self, words: List[str], mode: str = 'stress'):
        lower_words = [word.lower() for word in words]
        tokens = self.pad_sequence(self.tokenizer.transform(lower_words))

        if self._cuda:
            sequences = torch.tensor(tokens, dtype=torch.long).cuda()
        else:
            sequences = torch.tensor(tokens, dtype=torch.long)

        preds = self.model(sequences)
        indices = torch.argmax(preds, dim=1)

        if mode == 'stress':
            stress = chr(769)
            shift = 1
        elif mode == 'asterisk':
            stress = "*"
            shift = 1
        elif mode == 'plus':
            stress = "+"
            shift = 0
        else:
            raise ValueError(f"Wrong `mode`={mode}")

        return [word[:index + shift] + stress + word[index + shift:] for word, index in zip(words, indices)]

    def process(self, text: str, mode: str = 'stress'):
        words = tokenize(text)
        words_list, index_list = zip(*words)

        stressed_list = self.predict(words_list, mode = mode)
        stressed_words = zip(stressed_list, index_list)

        stressed_text = detokenize(text, stressed_words)
        
        return stressed_text

    @staticmethod
    def load_dict(path: str):
        dictionary = []

        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    stressed_word = line.strip()
                    unstressed_word = remove_accents(stressed_word)
                    stressed_word = replace_accents(stressed_word)

                    r1 = stressed_word.strip().replace('i', 'і')
                    r2 = unstressed_word.strip().replace('i', 'і')

                    dictionary.append((r2, r1))
                except ValueError:
                    continue

        return dictionary