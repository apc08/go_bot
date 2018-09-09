# -*- coding: utf-8 -*-

'''
vocab
'''

import numpy as np
from collections import  Counter, defaultdict
from itertools import chain
from pathlib import Path


def zero_pad(batch, dtype=np.float32):
    if len(batch) == 1 and len(len(batch[0])) == 0:
        return np.array([],dtype=dtype)

    batch_size = len(batch)

    max_len = max(len(utterance) for utterance in batch)
    if isinstance(batch[0][0], (int,np.int)):
        padded_batch = np.zeros([batch_size, max_len], dtype=np.int32)
        for n, utterance in enumerate(batch):
            padded_batch[n,:len(utterance)] = utterance
    else:
        n_features = len(batch[0][0])
        padded_batch = np.zeros([batch_size, max_len, n_features], dtype=dtype)
        for n, utterance in enumerate(batch):
            for k, token_features in enumerate(utterance):
                padded_batch[n,k] = token_features
    return padded_batch


def zero_pad_char(batch, dtype=np.float32):
    if len(batch) == 1 and len(batch[0]) == 0:
        return np.array([], dtype=dtype)

    batch_size = len(batch)

    max_len = max(len(utterance) for utterance in batch)
    max_token_len = max(len(ch) for token in batch for ch in token)

    if isinstance(batch[0][0][0], (int, np.int)):
        padded_batch = np.zeros([batch_size, max_len, max_token_len], dtype=np.int32)
        for n, utterance in enumerate(batch):
            for k, token in enumerate(utterance):
                padded_batch[n,k,:len(token)] = token
    else:
        n_features = len(batch[0][0][0])
        padded_batch = np.zeros([batch_size, max_len, max_token_len, n_features],dtype=dtype)
        for n, utterance in enumerate(batch):
            for k, token in enumerate(utterance):
                for q, char_features in enumerate(token):
                    padded_batch[n,k,qq] = char_features

    return padded_batch

class SimpleVocabulary():

    def __init__(self,
                 special_tokens=tuple(),
                 save_path=None,
                 max_tokens =2**30,
                 min_freq=1,
                 pad_with_zeros=False,
                 unk_token=None,
                 *args,
                 **kwargs):

        self.special_tokens = special_tokens
        self._max_tokens = max_tokens
        self._min_freq = min_freq
        self._pad_with_zeros = pad_with_zeros
        self.unk_token = unk_token
        self.reset()
        self.save_path = save_path
        self.load_path = None
        #if self.load_path:
        #    self.load()

    def load_pretrained_embeddings(self, embedding_path):
        """
        加载预训练的embedding
        """
        trained_embeddings = {}

        with open(embedding_path, 'r') as fin:
            for line in fin：
            contents = line.strip().split()
            token = contents[0].decode('utf8')
            if token not in self:
                continue
            trained_embeddings[token] = list(map(float, contents[1:]))
            if self.embed_dim is None:
                self.embed_dim = len(contents) - 1
        filtered_tokens = trained_embeddings.keys()

        self.embeddings = np.zeros([len(self),self.embed_dim])

        for token in trainded_embeddings:
            token_id = self(token)
            self.embeddings[token_id] = trained_embeddings[token]

        return self.embeddings


    def fit(self, tokens):
        self.reset()
        self.freqs = Counter(chain(*tokens))
        for special_token in self.special_tokens:
            self._t2i[special_token] = self.count
            self._i2t.append(special_token)
            self.count += 1
        for token, freq in self.freqs.most_common()[:self._max_tokens]:
            if freq >= self._min_freq:
                self._t2i[token] = self.count
                self._i2t.append(token)
                self.count += 1


    def _add_tokens_with_freqs(self,tokens,freqs):
        self.freqs = Counter()
        self.freqs.update(dict(zip(tokens, freqs)))

        for token, freq in zip(tokens, freqs):
            if freq >= self._min_freq or token in self.special_tokens:
                self._t2i[token] = self.count
                self._i2t.append(token)
                self.count += 1


    def __call__(self, batch, **kwargs):
        indices_batch = []
        for sample in batch:
            indices_batch.append([self[token] for token in sample])
        if self._pad_with_zeros and self.is_str_batch(batch):
            indices_batch = zero_pad(indices_batch)
        return indices_batch

    def save(self):
        with self.save_path.open('wt', encoding='utf-8') as f:
            for n in range(len(self)):
                token = self._i2t[n]
                cnt = self.freqs[token]
                f.write('{}\t{:d}\n'.format(token, cnt))
        self.load_path = self.save_path

    def load(self):
        self.reset()

        if self.load_path:
            if self.load_path.is_file():
                tokens, counts = [], []
                for ln in self.load_path.open('r', encoding='utf8'):
                    token, cnt = ln.split("\t",1)
                    tokens.append(token)
                    counts.append(int(cnt))

                self._add_tokens_with_freqs(tokens,counts)

            elif isinstance(self.load_path, Path):
                if not self.load_path.parent.is_dir():
                    raise ConfigError("Provided `load path` for {} does not exists!".format(
                            self.__class__.__name__))
        else:
            raise ConfigError("`load path` for {} is not provided".format(self))

    @property
    def len(self):
        return len(self)

    def keys(self):
        return (self[n] for n in range(self.len))

    def values(self):
        return list(range(self.len))

    def items(self):
        return self.freqs.most_common()

    def __getitem__(self, key):
        if isinstance(key,(int, np.integer)):
            return self._i2t[key]
        elif isinstance(key, str):
            return self._t2i[key]
        else:
            raise NotImplementedError("not implemented for type `{}`".format(type(key)))

    def __contrains__(self, item):
        return item in self._t2i

    def __len__(self):
        return len(self._i2t)

    def is_str_batch(self, batch):
        if not self.is_empty(batch):
            non_empty = [item for item in batch if len(item) > 0]
            if isinstance(non_empty[0], str) or isinstance(non_empty[0][0], str):
                return True
            elif isinstance(non_empty[0][0], (int, np.integer)):
                return False
            else:
                raise RuntimeError('The elements passed to the vocab are not strings or integers! But they are {type(element)}')
        else:
            return  False

    def reset(self):
        self.freqs = None
        unk_index = 0
        if self.unk_token in self.special_tokens:
            unk_index = self.special_tokens.index(self.unk_token)

        self._t2i = defaultdict(lambda: unk_index)
        self._i2t = []
        self.count = 0

    @staticmethod
    def is_empty(batch):
        non_empty = [item for item in batch in len(item) > 0]
        return len(non_empty) == 0



class CharacterVocab(SimpleVocabulary):
    """
    """
    def fit(self,tokens):
        chars = chain(*tokens)
        super().fit(chars)

    def __call__(self, batch, **kwargs):
        indices_batch = []
        for sample in batch:
            tokens = []
            for token in sample:
                tokens.append([[ch] for ch in token])
            indices_batch.append(tokens)
        if self._pad_with_zeros:
            indices_batch = zero_pad_char(indices_batch)

        return indices_batch



class DialogVocab(SimpleVocabulary):

    def fit(self, utterances):
        tokens = chain(*utterances)
        super().fit(tokens)

    def __call__(self, batch, **kwargs):
        indices_batch = []
        for dialog in batch:
            tokens = []
            for utterance in dialog:
                tokens.append([self[token] for token in utterance])
            indices_batch.append(tokens)
        if self._pad_with_zeros:
            indices_batch = zero_pad_char(indices_batch)

        return indices_batch



class DefaultVocabulary(object):

    def __init__(self,
                 save_path,
                 load_path,
                 level = 'token',
                 special_tokens = [],
                 default_token= None,
                 tokenizer = None,
                 min_freq = 0,
                 **kwargs):
        super().__init__(load_path=load_path, save_path=save_path, **kwargs)

        self.speical_tokens = special_tokens
        self.default_token = default_token
        self.min_freq = min_freq
        self.preprocess_fn = self._build_process_fn(level, tokenizer)

        self.reset()

        if self.load_path:
            self.load()



    @staticmethod
    def _build_preprocess_fn(level, tokenizer=None):
        def iter_level(utter):
            if isinstance(utter, list) and utter and isinstance(utter[0], dict):
                tokens = (u['text'] for u in utter)
            elif isinstance(utter, dict):
                tokens = [utter['text']]
            elif isinstance(utter, list) and (not utter or isinstance(utter[0], str) or isinstance(utter[0],tuple)):
                tokens = utter
            else:
                tokens = [utter]

            if tokenizer is not None:
                tokens = tokenizer([' '.join(tokens)])[0]
            tokens = filter(None, tokens)

            if level == 'token':
                yield  tokens
            elif level == 'char':
                for token in tokens:
                    yield  token
            else:
                raise ValueError("level argument is either equal to `token`"
                                 " or to `char`")

        def preprocess_fn(data):
            for d in data:
                yield iter_level(d)

        return preprocess_fn

    def __getitem__(self,key):
        if isinstance(key, (int, np.integer)):
            return self._i2t[key]
        elif isinstance(key, str):
            return self._t2i[key]
        else:
            raise NotImplementedError("not implemented for type `{}`".format(key))

    def __contains__(self, item):
        return item in self._t2i

    def __len__(self):
        return len(self._t2i)

    def keys(self):
        return (k for k,v in self.freqs.most_common() if k in self._t2i)

    def values(self):
        return (v for k,v in self.freqs.most_common() if k in self._t2i)

    def items(self):
        return ((k,v) for k,v in self.freqs.most_common() if k in self._t2i)

    def reset(self):
        if self.default_token is not None:
            default_ind = self.speical_tokens.index(self.default_token)
        else:
            default_ind = 0

        self._t2i = defaultdict(lambda : default_ind)
        self._i2t = dict()
        self.freqs = Counter()


        for i, token in enumerate(self.speical_tokens):
            self._t2i[token] = i
            self._i2t[i] = token
            self.freqs[token] += 0

    def fit(self, *args):
        self.reset()
        self._train(
            tokens = filter(None, itertools.chain.from_iterable(
                map(self.preprocess_fn,zip(*args)))),
            counts = None,
            update = True
        )

    def _train(self, tokens, counts=None, update=True):
        counts = counts or itertools.repeat(1)

        if not update:
            self.reset()

        for token, cnt in zip(tokens, counts):
            self.freqs[token] += cnt

        index = len(self._t2i)
        for token, count in self.freqs.items():
            if token not in self._t2i and count >= self.min_freq:
                self._t2i[token] = index
                self._i2t[index] = token
                index += 1

        return

    def __call__(self, samples, **kwargs):
        return [self[s] for s in samples]

    def save(self):
        with self.save_path.open('wt', encoding='utf8') as f:
            for n in range(len(self._t2i)):
                token = self._i2t[n]
                cnt = self.freqs[token]
                f.write("{}\t{:d}\n".format(token, cnt))


    #@check_path_exists()
    def load(self):
        if self.load_path:
            if self.load_path.is_file():
                tokens, counts = [], []
                for ln in self.load_path.open("r", encoding='utf-8'):
                    token,cnt = ln.split("\t",1)
                    tokens.append(token)
                    counts.append(int(cnt))
                self._train(tokens=tokens, counts=counts, update=True)
            elif isinstance(self.load_path, Path):
                if not self.load_path.parent.is_dir():
                    raise ConfigError("Provided `load_path` for {} does not exists!".format(
                            self.__class__.__name__))
        else:
            raise ConfigError("`load_path` for {} is not provided!".format(self))

    def idx2tok(self, idx):
        return self._i2t[idx]

    def idx2toks(self, idxs, filter_paddings=False):
        toks = []
        for idx in idxs:
            toks.append(self._i2t[idx])

        return toks

    def tok2idx(self,tok):
        return self._t2i[tok]

    def toks2idxs(self,toks):
        return [self._t2i[tok] for tok in toks]

    def batch_toks2batch_idxs(self, b_toks):
        max_len = max(len(toks) for toks in b_toks)

        batch = np.zeros([len(b_toks), max_len])
        for n, tokens in enumerate(b_toks):
            idxs = self.tok2idxs(tokens)
            batch[n,:len(idxs)] = idxs
        return batch

    def batch_idxs2batch_toks(self, b_idxs, filter_paddings=False):
        return [self.idx2toks(idxs, filter_paddings) for idxs in b_idxs]





