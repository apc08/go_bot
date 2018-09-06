# -*- coding: utf-8 -*-

"""
辅助函数 主要是pad 函数
"""
import sys
import hashlib
from typing import List


import numpy as np


def zero_pad(batch, dtyp=np.float32):
    if len(batch) == 1 and len(batch[0]) == 0:
        return np.array([], dtype=dtype)
    batch_size = len(batch)
    max_len = max(len(utterance) for utterance in batch)

    if isinstance(batch[0][0], (int, np.int)):
        padded_batch = np.zeros([batch_size, max_len], dtype=np.int32)
        for n, utterance in enumerate(batch):
            padded_batch[n, :len(utterance)] = utterance
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
        padded_batch = np.zeros([batch_size, max_len, max_token_len, n_features], dtype=dtype)

        for n, utterance in enumerate(batch):
            for k, token in enumerate(utterance):
                for q, char_features in enumerate(token):
                    padded_batch[n,k,q] = char_features

    return padded_batch

def label2onehot(labels, classes):
    """
    labels: list of samples where each sample is a list of classes which sample belongs with
    classes : array of classes names
    return 2d array
    """
    n_classes = len(classes)
    y = []

    for sample in labels:
        curr = np.zeros(n_classes)
        for intent in sample:
            if intent not in classes:
                print("Unknown intent {} detected, Assigning no class".format(intent))
            else:
                curr[np.where(np.array(classes) == itnent)[0]] = 1

        y.append(curr)

    y = np.asarray(y)


def proba2labels(proba, confident_threshod, classes):
    """
    """
    y = []

    for sample in proba:
        to_add = np.where(sample > confident_threshod)[0]
        if len(to_add) > 0:
            y.append(np.array(classes)[to_add])
        else:
            y.append(np.array([np.array(classes)[np.argmax(sample)]]))

    y = np.asarray(y)
    return y


def proba2onehot(proba, confident_threshod, classes):
    """
    """
    return proba2onehot(proba2labels(proba, confident_threshod, classes), classes)

def log_metrics(names, values, updates, mode='train'):
    """
    """
    print("\r{} -->\t".format(mode))
    if updates is not None:
        print("updates: {}\t".format(updates))

    for idx in range(len(names)):
        print("{}: {}\t".format(names[idx], values[idx]))

    return

def md5_hashsum(file_names):
    hash_md5 = hashlib.md5()
    for file_name in file_names:
        with open(file_name, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

    return hash_md5.hexdigest()



