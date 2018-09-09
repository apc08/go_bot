# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav


def bilinear_attetion(question_rep, passage_repres,  passage_mask):
    """
    常规的attention 机制

    args:
        question_rep : [batch_size, hidden_size]
        passage_repres : [batch_size, sequence_length, hidden_size]
        passage_mask : [batch, sequence_length]

    returns:
        passage_rep : [batch_size, hidden_size]
    """
    hidden_size = question_rep.get_shape()[1]
    # [hidden_size, hidden_size]
    W_bilinear = tf.get_variable("W_bilinear", shape=[hidden_size, hidden_size], dtype=tf.float32)

    # [batch_size, hidden_size]
    question_rep = tf.matmul(question_rep, W_bilinear)

    # [batch_size, 1, hidden_size]
    question_rep = tf.expand_dims(question_rep, 1)

    # [batch_size, seq_length]
    alpha = tf.nn.softmax(tf.reduce_sum(question_rep * passage_repres, axis=2))
    alpha = alpha * passage_mask
    alpha = alpha / tf.reduce_sum(alpha, axis=-1, keep_dims=True)

    # [batch_size, hidden_size]
    passage_rep = tf.reduce_sum(passage_repres * tf.expand_dims(alpha, axis=-1), axis=1)

    return passage_rep




def general_attention(key, context, hidden_size, projected_align=False):
    """
    key: [None, None, Key_size]   # batch, turn_num, key_dim
    context: [None, None, max_num_tokens, token_size]   # batch turn_num sent_len token_dim
    hidden_size,
    projected_align: Using bidirectional lstm for hidden representation of context.
       如果是true 在输入和attention 机制layer 中间使用 bi lstm
       如果不是 不使用 bi-directiona rnn

    output: [None, None, hidden_size]   batch turn_num , hidden_size
    """
    if hidden_size % 2 != 0:
        raise ValueError("hidden size must be devideable ty two")
    batch_size = tf.shape(context)[0]
    max_num_tokens, token_size = context.get_shape().as_list()[-2:]
    # shape [batch_size * turn_num , sent_len, token_dim]
    r_context = tf.reshape(context, shape=[-1, max_num_tokens, token_size])

    # projected_key: [None, None, hidden]
    projected_key = \
        tf.layers.dense(key, hidden_size, kernel_initializer=xav())

    # [batch_size * turn_num, hidden_size, 1]   # 意图 和 intent key 和每个字
    # 相乘 获得对应的attention 值
    r_projected_key = tf.reshape(projected_key, shape=[-1, hidden_size,1])

    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)

    (output_fw, output_bw), states = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                        cell_bw=lstm_bw_cell,
                                        inputs=t_context,
                                        dtype=tf.flaot32)

    # bilstm output [bath*sent_len, max_num_tokens, hidden_size]
    bilstm_output = tf.concat([output_fw, output_bw],-1)

    attn = tf.nn.softmax(tf.matmul(bilstm_output, r_projected_key), dim=1)

    if projected_align:
        print('使用 projected attetnion alignment')
        t_context = tf.transpose(bilstm_output, [0,2,1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size,-1,hidden_size])
    else:
        print("不使用映射机制")
        t_context = tf.transpose(r_context,[0,2,1])

        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size,-1, token_size])

    return output


def light_general_attention(key, context, hidden_size, projected_align):
    """
    参数定义同上
    """

    batch_size = tf.shape(context)[0]
    max_num_tokens, token_size = context.get_shape().as_list()[-2:]
    r_context = tf.reshape(context, shape=[-1, max_num_tokens, token_size])

    projected_key = tf.layers.dense(key,hidden_size, kernel_initializer=xav())
    r_projected_key = tf.reshape(projected_key, shape=[-1, hidden_size,1])

    # 直接使用 线性映射， 不使用 bidirection
    projected_context = \
        tf.layers.dense(r_context, hidden_size, kernel_initializer=xav())

    attn = tf.nn.softmax(tf.matmul(projected_context, r_projected_key), dim=1)

    if projected_align:
        t_context = tf.transpose(projected_context, [0,2,1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size,-1,hidden_size])

    else:
        print("Using without projected attention alignment")
        t_context = tf.transpose(t_context, [0,2,1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size, -1, token_size])

    return output



#def cs_general_attention(key, context, hidden_size, depth, projected_align=False):
#    """
#    定义 luong attent 机制
#    参数定义同上
#    """
#    if hidden_size % 2 != 0:
#        raise ValueError("hidden size must be dividable ty two")
#
#    key_size = tf.shape(key)[-1]
#    batch_size = tf.shape(context)[0]
#    max_num_tokens, token_size = context.get_shape().as_list()[-2:]
#    r_context = tf.reshape(context, shape=[-1, max_num_tokens, token_size])
#
#    projected_context = tf.layers.dense(r_context, token_size,
#                                        kernel_initializer=xav(),
#                                        name='projected_context')
#
#    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size // 2)
#    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size // 2)
#
#    (output_fw, output_bw), states = \
#        tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
#                                        cell_bw=lstm_bw_cell,
#                                        inputs=projected_context,
#                                        dtype=tf.float32)
#
#    # bilstm output [-1, max_num_tokens, hidden_size]
#    bilstm_output = tf.concat([output_fw, output_bw], -1)
#    h_state_for_sketch = bilstm_output
#
#    if projected_align:
#        print('Using projected attention alignment')
#        h_state_for_attn_alignment = bilstm_output
#        aligned_h_state = csoftmax_attention.attention_gen_block(
#            h_state_for_sketch, h_state_for_attn_alignment, key, depth)
#
#        output = \
#            tf.reshape(aligned_h_state, shape=[batch_size, -1, depth * hidden_size])
#
#    else:
#        print("Using without projected attention alignment")
#        h_state_for_attn_alignment = projected_context
#        aligned_h_state = csoftmax_attention.attention_gen_block(
#            h_state_for_sketch, h_state_for_attn_alignment, key, depth)
#        output = \
#            tf.reshape(aligned_h_state, shape=[batch_size, -1, depth * token_size])
#
#    return output
#
#def bahdanau_attention(key, context, hidden_size, projected_align=False):
#    """
#    同上
#    """
#    pass
#
#
#
#

def bahdanau_attention(key, context, hidden_size, projected_align=False):
    """
    参数输出同上
    """
    if hidden_size % 2 != 0:
        raise ValueError("hidden size must be dividable by two")

    batch_size = tf.shape(context)[0]
    max_num_tokens, token_size = context.get_shape().as_list()[-2:]
    # [batch_size * turn_num, max_token_num, token_size]
    r_context = tf.reshape(context, shape=[-1, max_num_tokens, token_size])

    # projected key [None, None, hidden_size]  batch , turn_num, hidden_size
    projected_key = tf.layers.dense(key, hidden_size, kernel_initializer=xav())

    # [batch * turn_num, max_num_tokens, hidden_size]
    r_projected_key = \
        tf.tile(tf.reshape(projected_key, shape=[-1, 1, hidden_shape]),
                [1, max_num_tokens, 1])

    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)

    (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                     cell_bw = lst_bw_cell,
                                                                     inputs= r_context,
                                                                     dtype=tf.float32)

    # bilstm_output [-1, self.max_tokens, _n_hidden]
    bilstm_output = tf.concat([output_fw, output_bw], -1)
    concat_h_state = tf.concat([r_projected_key, output_fw, output_bw],-1)

    projected_state = \
        tf.layers.dense(concat_h_state, hidden_size, use_bias=False,
                        kernel_initializer=xav())

    score = tf.layers.dense(tf.tanh(projected_state), units=1, use_bias=False,
                            kernel_initializer=xav())

    attn = tf.nn.softmax(score, dim=1)

    if projected_align:
        print("Using projected attention alignment")
        t_context = tf.transpose(bilstm_output, [0,2,1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size,-1,hidden_size])

    else:
        print("Using without projeced attention alignment")
        t_context = tf.transpose(r_context, [0,2,1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size,-1,token_size])

    return output


def light_bahdanau_attention(key, context, hidden_size, projected_align=False):
    """
    参数输出维度同上
    """
    batch_size = tf.shape(context)[0]
    max_num_tokens, token_size = context.get_shape().as_list()[-2:]
    # [batch_size * turn_num, max_token_num, token_size]
    r_context = tf.reshape(context, shape=[-1, max_num_tokens, token_size])

    # projected key: [None, None, hidden_size]       batch, turn_num , hidden_dim
    projected_key = tf.layers.dense(key, hidden_size, kernel_initializer=xav())
    # batch * turn_num , max_token_num, hidden_dim
    r_projected_key = \
        tf.tile(tf.reshape(projected_key, shape=[-1,1,hidden_size]),
                [1, max_num_tokens, 1])

    # projected context  [None, max_num_tokens, hidden_size]
    projected_context = \
        tf.layers.dense(r_context, hidden_size, kernel_initializer=xav())
    concat_h_state = tf.concat([projected_context, r_projected_key], -1)

    projected_state = \
        tf.layers.dense(concat_h_state, hidden_size, use_bias=False,
                        kernel_initializer=xav())
    score = \
        tf.layers.dense(tf.tanh(projeced_state), units=1, use_bais=False,
                        kernel_initializer=xav())
    attn = tf.nn.softmax(score, dim=1)

    if projected_align:
        print("Using projected attention alignment")
        t_context = tf.transpose(projected_context, [0,2,1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size,-1,hidden_size])
    else:
        print("Using without projected attention alignment")
        t_context = tf.transpose(r_context, [0,2,1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size,-1,token_size])

    return output











