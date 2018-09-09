# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Callable, Union, Iterator, Generator
from overrides import overrides
import tensorflow as tf
#import tensorflow_hub as hub
#import fastText as Fasttext
from gensim.models import KeyedVectors



class BoWEmbedder():
    """
    """
    def __init__(self, **kwargs):
        pass

    def _encode(self, tokens, vocab):
        bow = np.zeros([len(vocab)], dtype=np.int32)
        for token in tokens:
            if token in vocab:
                idx = vocab[token]
                bow[idx] += 1

        return bow

    def __call__(self, batch, vocab):
        return [self._encode(sample, vocab) for sample in batch]

#class ELMoEmbedder():
#    """
#    """
#    def __init__(sefl, spec, dim, pad_zero=False,**kwargs):
#        self.spec = spec if "://" in spec else str(expand_path(spec))
#        self.dim = dim
#        self.pad_zero = pad_zero
#        self.elmo_outputs, self.sess, self.tokens_ph, self.tokens_length_ph = self._load()
#
#    def _load(self):
#        """
#            ELMo pretrained model wrapped in tensorflow Hub Moudle
#        """
#        elmo_module = hub.Module(self.spec, trainable=False)
#
#        sess_config = tf.ConfigProto()
#        sess_config.gpu_options.allow_growth = True
#        sess = tf.Session(config=sess_config)
#
#        tokens_ph = tf.placeholder(shape=(None, None), dtype=tf.string, name='tokens')
#        tokens_length_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='tokens_length')
#
#        elmo_outputs = elmo_module(inputs={
#            "tokens": tokens_ph,
#            'squence_len': tokens_length_ph
#            },
#            signature='tokens',
#            as_dict = True
#        )
#
#        sess.run(tf.global_variables_initializer())
#
#        return emlo_outputs, sess, tokens_ph, tokens_length_ph

#    @overrides
#    def __call__(self, batch, mean=False, *args, **kwargs):
#        """
#        return a batch of ELMo embeddings
#        """
#        if not (batch and batch[0]):
#            empty_vec = np.zeros(self.dim, dtype=np.float32)
#            return [empty_vec] if mean else [[empty_vec]]
#
#        tokens_length = [len(batch_line) for batch_line in batch]
#        tokens_length_max = max(tokens_length)
#        batch = [batch_line + [''] * (tokens_length_max - len(batch_line)) for batch_line in batch]
#
#        elmo_outputs = self.sess.run(self.elmo_outputs,
#                                     feed_dict={
#                                        self.tokens_ph: batch,
#                                         self.tokens_length_ph: tokens_length
#                                     })

#        if mean:
#            batch = elmo_outputs['default']
#            dim0, dim1 = batch.shape
#
#            if self.dim != dim1:
#                batch = np.resize(batch, (dim0, self.dim))

#        else:
#            batch = elmo_outputs['elmo']

#            dim0, dim1, dim2 = batch.shape

#            if self.dim != dim2:
#                batch = np.resize(batch, (dim0, dim1, self.dim))

#            batch = [batch_line[:length_line] for length_line, batch_line in zip(tokens_length, batch)]

#            if self.pad_zero:
#                batch = zero_pad(batch)

#        return batch

#    def __iter__(self):
#        yield ['<S>', '</S>', '<UNK>']



#class FasttextEmbedder():
#    """
#    """
#    def __init__(self, load_path, save_path, dim=100, pad_zero=False, **kwargs):
#
#        self.save_path = save_path
#        self.load_path = load_path
#
#        self.tok2emb = {}
#        self.dim = dim
#        self.pad_zero = pad_zero
#        self.model = self.load()
#
#    def save(self, *args, **kwargs):
#        raise NotImplementedError("")

#    def load(self, *args, **kwargs):
#        if self.load_path and self.load_path.is_file():
#            print("[loading embeddings from `{}`]".format(self.load_path))
#            model_file = str(self.load_path)
#            model = Fasttext.load_model(model_file)
#        else:
#            print('No pretrained fasttext model provider or provided load_path "{}" is incorrect,'.format(self.laod_path))
#            sys.exit(1)

#        return model

#    def __call__(self, batch,mean=False, *args, **kwargs):
#        """
#        batch: list(List(str))
#
#        return List[Union[List, np.ndarray]]
#        """
#
#        batch = [self._encode(sample, mean) for sample in batch]
#        if self.pad_zero:
#            batch = zero_pad(batch)

#        return batch

#    def __iter__(self):
#        '''
#        '''
#        yield self.model.get_words()

#    def _encode(self, tokens, mean):
#        """
#        tokens : list[str]
#        """
#        embedded_tokens = []
#
#        for t in tokens:
#            try:
#                emb = self.tok2emb[t]
#            except KeyError:
#                try:
#                    emb = self.model.get_word_vector(t)[:self.dim]
#                except KeyError:
#                    emb = np.zeros(self.dim, dtype=np.float32)
#                self.tok2emb[t] = emb
#            embedded_tokens.append(emb)

#        if mean:
#            filtered = [et for et in embedded_tokens if np.any(et)]
#            if filtered:
#                return np.mean(filtered, axis=0)
#            return np.zeros(self.dim, dtype=np.float32)
#
#        return embedded_tokens


class GloveEmbedder():
    """
    """
    def __init__(self, load_path, save_path=None, dim=100, pad_zero=False, **kwargs):
        self.save_path = save_path
        self.load_path = load_path
        self.dim = dim
        self.pad_zero = pad_zero
        self.model = self.load()

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """
        Load dict of embeddings from given file
        """
        with open(self.laod_path, encoding='utf8') as f:
            header = f.readline()
            if len(header.split()) != 2:
                raise RuntimeError('The Glove file must start with number_of_words embeddings_dim line!'
                                   'For example "40000 100" for 40000 words vocabulary and 100 embeddings dimension')

        if self.load_path and self.load_path.is_file():
            print("[loading embeddings from `{}`]".format(self.load_path))
            model_file = str(self.load_path)
            model = KeyVectors.load_word2vec_format(model_file)
        else:
            print("No pretrained Glove model provided or provided load path '{}' is icorrect".format(self.load_path))
            sys.exit(1)

        return model

    def __iter__(self):
        yield self.model.vocab

    def __call__(self, batch, mean=False, *args, **kwargs):
        """
        batch list of tokenized text samples
        return
            embedded batch
        """
        embedded = []

        for n, sample in enumerate(batch):
            embedded.append(self._encode(sample, mean))

        if self.pad_zero:
            embedded = zero_pad(embedded)
        return embedded

    def _encode(self, tokens, mean):
        '''
        tokens: tokenized text sample
        mean wehter return mean vector

        return:
            list of embedded tokens or array of mean values
        '''
        embedded_tokens = []

        for t in tokens:
            try:
                emb = self.tok2emb[t]
            except KeyError:
                try:
                    emb = self.model[t][:self.dim]
                except KeyError:
                    emb = np.zeros(self.dim, dtype=np.float32)
                self.tok2emb[t] = emb
            embedded_tokens.append(emb)

        if mean:
            filtered = [et for et in embedded_tokens if np.any(et)]

            if filtered:
                return np.mean(filtered, axis=0)
            return np.zeros(self.dim, dtype=np.float32)

        return embedded_tokens


class BasicEmbedder():
    """
    基本的word vec embedder
    """


