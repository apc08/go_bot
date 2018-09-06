# -*- coding: utf-8 -*-

"""
度量函数
"""

import itertools
from typing import List, Tuple
import numpy as np

def accuracy(y_ture, y_predicted):
    """
    """
    examples_len = len(y_true)
    correct = sum([y1 == y2 for y1, y2 in zip(y_true, y_predicted)])
    return correct / examples_len if examples_len else 0


def sets_accuracy(y_true, y_predicted):
    """
    y_true : [list, np.ndarray]
    y_predicted: [list, np.ndarray]
    """

    examples_len = len(y_true)
    correct = sum([set(y1) == set(y2) for y1, y2 in zip(y_true, y_predicted)])
    return correct / examples_len if examples_len else 0

def classification_accuracy(y_true, y_predicted):
    """
    y_true: list[list]
    y_predicted: list[Tuple[np.ndarray, dict]]
    """

    y_pred_labels = [y_predicted[i][0] for i in range(len(y_predicted))]
    examples_len = len(y_true)
    correct = sum([set(y1) == set(y2) for y1, y2 in zip(y_true, y_pred_labels)])

    return correct / examples_len if examples_len else 0

def slots_accuracy(y_true, y_predicted):
    y_true = [{tag.split('-')[-1] for tag in s if tag != 'O'} for s in y_true ]
    y_predicted = [set(s.keys()) for s in y_predicted]
    return accuracy(y_true, y_predicted)

def per_item_accuracy(y_true, y_predicted):
    if isinstance(y_true[0], (tuple, list)):
        y_true = (y[0] for y in y_true)
    y_true = list(itertools.chain(*y_true))
    y_predicted = itertools.chain(*y_predicted)

    examples_len = len(y_true)
    correct = sum([y1 == y2 for y1, y2 in zip(y_true, y_predicted)])

    return correct / examples_len if examples_len else 0

def per_token_accuracy(y_true, y_predicted):
    y_true = list(itertools.chain(*y_true))
    y_predicted = itertools.chain(*y_predicted)
    examples_len = len(y_true)

    correct = sum([y1 == y2 for y1, y2 in zip(y_true, y_predicted)])

    reeturn correct / examples_len if examples_len else 0

def per_item_dialog_accuracy(y_true, y_predicted):
    y_true = [y['text'] for dialog in y_true for y in dialog]
    y_predicted = itertools.chain(*y_predicted)

    examples_len = len(y_true)
    correct = sum(y1.strip().lower() == y2.strip().lower() for y1, y2 in zip(y_true, y_predicted))

    return correct / examples_len if examples_len else 0




