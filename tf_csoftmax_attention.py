# -*- coding: utf-8 -*-

"""
注意力辅助函数
"""

import tensorflow as tf

def csoftmax_for_slice(input):
    """
    args:
        [input tensor, cumulative attention]
    returns:
        output: a list of [csoftmax results, masks]
    """
    [ten, u] = input
    shape_t = ten.shape
    shape_u = u.shape

    ten -= tf.reduce_mean(ten)
    q = tf.exp(ten)
    active = tf.ones_like(u, dtype=tf.int32)
    mass = tf.constant(0, dtype=tf.float32)
    found = tf.constant(True, dtype=tf.bool)




