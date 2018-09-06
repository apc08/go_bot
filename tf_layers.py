# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
from typing import List


INITIALIZER = tf.orthogonal_initializer
# INITIALIZER = xavier_initializer


def stacked_cnn(units,
                n_hidden_list,  # []
                filter_width=3,
                use_batch_norm = False,
                use_dilation=False,
                training_ph = None,
                add_l2_losses=False):
    """
    units : [batch, n_tokens, n_features]

    """
    for n_layer, n_hidden in enumerate(n_hidden_list):
        if use_dilation:
            dilation_rate = 2 ** n_layer
        else:
            dilation_rate = 1

        units = tf.layers.conv1d(units,
                                 n_hidden,
                                 filter_width,
                                 padding='same',
                                 dilation_rate=dilation_rate,
                                 kernel_initializer=INITIALIZER(),
                                 kernel_regularizer=tf.nn.l2_loss)

        if use_batch_norm:
            assert training_ph is not None
            units = tf.layers.batch_normalization(units, training=training_ph)
        units = tf.nn.relu(units)

    return units

def dense_convolutional_network(units,
                                n_hidden_list,
                                filter_width=3,
                                use_dilation=False,
                                use_batch_norm=False,
                                training_ph=None):
    """
    return  [None, n_tokens, n_hidden_list[-1]]
    """
    units_list = [units]

    for n_layer, n_filters in enumerate(n_hidden_list):
        total_units = tf.concat(units_list, axis=1)
        if use_dilation:
            dilation_rate = 2 ** n_layer
        else:
            dilation_rate = 1

        units = tf.layers.conv1d(total_units,
                                 n_filters,
                                 filter_width,
                                 dilation_rate=dilation_rate,
                                 padding='same',
                                 kernel_initializer=INITIALIZER())

        if use_batch_norm:
            units = tf.layers.batch_normalization(units, training=training_ph)

        units = tf.nn.relu(units)
        units_list.append(units)

    return units


def bi_rnn(units,
           n_hidden,
           cell_type='gru',
           seq_lengths=None,
           trainable_initial_states=False,
           use_peepholes=False,
           name='Bi-'):
    """

    return [None, n_tokens, n_hidden_list[-1]]
    [Bx2*H]
    """
    with tf.variable_scope(name + '_' + cell_type.upper()):
        if cell_type == 'gru':
            forward_cell = tf.nn.rnn_cell.GRUCell(n_hidden, kernel_initializer=INITIALIZER())
            backward_cell = tf.nn.rnn_cell.GRUCell(n_hidden, kernel_initializer=INITIALIZER())

            if trainable_initial_states:
                initial_state_fw = tf.tile(tf.get_variable('init_fw_h',[1, n_hidden]), (tf.shape(units)[0],1))
                initial_state_bw = tf.tile(tf.get_variable('init_bw_h',[1, n_hidden]), (tf.shape(units)[0],1))
            else:
                initial_state_fw = initial_state_bw = None

        elif cell_type == 'lstm':
            forward_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=use_peepholes, initializer=INITIALIZER())
            backward_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=use_peepholes, initializer=INITIALIZER())

            if trainable_initial_states:
                initial_state_fw = tf.nn.rnn_cell.LSTMStateTuple(
                    tf.tile(tf.get_variable('init_fw_c',[1,n_hidden]), (tf.shape(units)[0],1)),
                    tf.tile(tf.get_variable('init_fw_h',[1,n_hidden]), (tf.shape(units)[0],1)))

                initial_state_bw = tf.nn.rnn_cell.LSTMStateTuple(
                    tf.tile(tf.get_variable('init_bw_c', [1, n_hidden]), (tf.shape(units)[0],1)),
                    tf.tile(tf.get_variable('init_bw_h', [1, n_hidden]), (tf.shape(units)[0],1)))
            else:
                initial_state_fw = initial_state_bw = None

        else:
            raise RuntimeError('cell_type must be either "gru" or "lstm"s')
        (rnn_output_fw, rnn_output_bw), (fw,bw) = \
            tf.nn.bidirectional_dynamic_rnn(forward_cell,
                                            backward_cell,
                                            units,
                                            dtype=tf.float32,
                                            sequence_length=seq_lengths,
                                            initial_state_fw=initial_state_fw,
                                            initial_state_bw =initial_state_bw)

    kernels = [var for var in forward_cell.trainable_variables +
               backward_cell.trainable_variables if 'kernel' in var.name]

    for kernel in kernels:
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(kernel))

    return (rnn_output_fw, rnn_output_bw), (fw, bw)



def stacked_bi_rnn(units,
                   n_hidden_list,
                   cell_type='gru',
                   seq_lengths=None,
                   use_peepholes=False,
                   name='RNN_layer'):
    """
    units [name, tokens, n_features]

    """
    for n, n_hidden in enumerate(n_hidden_list):
        with tf.variable_scope(name + '_' + str(n)):
            if cell_type == 'gru':
                forward_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
                backward_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
            elif cell_type == 'lstm':
                forward_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=use_peepholes)
                backward_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=use_peepholes)
            else:
                raise RuntimeError('cell type must be either gru or lstm')

            (rnn_output_fw, rnn_output_bw), (fw, bw) = \
                tf.nn.bidirectional_dynamic_rnn(forward_cell,
                                                backward_cell,
                                                units,
                                                dtype=tf.float32,
                                                sequence_length=seq_lengths)
            units = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)

            if cell_type == 'gru':
                last_units = tf.concat([fw, bw], axis=1)
            else:
                (c_fw, h_fw), (c_bw,h_bw) = fw, bw
                c = tf.concat([c_fw, c_bw], axis=1)
                h = tf.concat([h_fw, h_bw], axis=1)
                last_units = (h, c)

    return units, last_units

def u_shape(units,
            n_hidden_list,   # list
            filter_width=7,
            use_batch_norm=False,
            training_ph=None):
    """
    units [None, n_token, n_features]
    return [None, n_tokens, n_hidden_list[-1]]
    """
    units_for_skip_conn = []

    conv_net_params = {'filter_width': filter_width,
                       'use_batch_norm': use_batch_norm,
                       'training_ph': training_ph}

    # Go down the rabbit hole
    for n_hidden in n_hidden_list:
        units = stacked_cnn(units, [n_hidden], **conv_net_params)
        units_skip_conn.append(units)
        units = tf.layers.max_pooling1d(units, pool_size=2, strides=2,padding='same')

    units = stacked_cnn(units, [n_hidden], **conv_net_params)

    # up to sun light

    for down_step, n_hidden in enumerate(n_hidden_list[::-1]):
        units = tf.expand_dims(units, axis=2)
        units = tf.layers.conv2d_transpose(units, n_hidden, filter_width, strides=(2,1),padding='same')
        units = tf.squeeze(units, axis=2)

        # skip connection
        skip_units = units_for_skip_conn[-(down_step + 1)]
        if skip_units.get_shape().as_list()[-1] != n_hidden:
            skip_units = tf.layers.dense(skip_units, n_hidden)
        units = skip_units + units

        units = stacked_cnn(units, [n_hidden], **conv_net_params)

    return units

def stacked_highway_cnn(units,
                        n_hidden_list,  # []
                        filter_width=3,
                        use_batch_norm=False,
                        use_dilation=False,
                        training_ph=None):
    """

    """

    for n_layer, n_hidden in enumerate(n_hidden_list):
        input_units = units

        if input_units.get_shape().as_list()[-1] != n_hidden:
            input_units = tf.layers.dense(input_units, n_hidden)
        if use_dilation:
            dilation_rate = 2 ** n_layer
        else:
            dilation_rate = 1

        units = tf.layers.conv1d(units,
                                 n_hidden,
                                 filter_width,
                                 padding='same',
                                 dilation_rate=dilation_rate,
                                 kernel_initializer=INITIALIZER())

        if use_batch_norm:
            units = tf.layers.batch_normalization(units, training_ph)

        sigmoid_gate = tf.layers.dense(input_units, 1, activation=tf.sigmoid, kernel_initializer=INITIALIZER())

        input_units = sigmoid_gate * input_units + (1 - sigmoid_gate) * units
        input_units = tf.nn.relu(input_units)

    units = input_units

    return units


def embedding_layer(token_indices=None,
                    token_embedding_matrix=None,
                    n_tokens=None,
                    token_embedding_dim=None,
                    name=None,
                    trainable=True):
    """

    return [B,T,E]    B - batch size  T - number of tokens E-token embedding dim
    """
    if token_embedding_matrix is not None:
        tok_mat = token_embedding_matrix
        if trainable:
            Warning('Matrix of embeddings is passed to the embedding_layer, '
                    'possibly there is a pre-trained embedding matrix. '
                    'Embeddings paramenters are set to Trainable!')

    else:
        #print(n_tokens, token_embedding_dim)
        tok_mat = np.random.randn(n_tokens, token_embedding_dim).astype(np.float32) / np.sqrt(token_embedding_dim)

    tok_emb_mat = tf.Variable(tok_mat, name=name, trainable=trainable)
    embedded_tokens = tf.nn.embedding_lookup(tok_emb_mat, token_indices)
    return embedded_tokens

def character_embedding_network(char_placeholder,
                                n_characters=None,
                                emb_mat=None,
                                char_embedding_dim=None,
                                filter_widths=(3,4,5,7),
                                highway_on_top=False):
    """
    char_placeholder: [B,T,C] B- batch_size  T- num of tokens  C- number of chaacters

    return [B,T,F]
    """
    if emb_mat is None:
        emb_mat = np.random.randn(n_characters, char_embedding_dim).astype(np.float32) / np.sqrt(char_embedding_dim)

    else:
        char_embedding_dim = emb_mat.shape[1]

    char_emb_var = tf.Variable(emb_mat, trainable=True)
    with tf.variable_scope("Char_Emb_Network"):
        # Character embedding layer
        c_emb = tf.nn.embedding_lookup(char_emb_var, char_placeholder)

        # character embedding network
        conv_results_list = []

        for filter_width in filter_widths:
            conv_results_list.append(tf.layers.conv2d(c_emb,
                                                      char_embedding_dim,
                                                      (1, filter_width),
                                                      padding='same',
                                                      kernel_initializer=INITIALIZER))
        units = tf.concat(conv_results_list, axis=3)
        units = tf.reduce_max(units, axis=2)
        if highway_on_top:
            sigmoid_gate = tf.layers.dense(units,
                                           1,
                                           activation=tf.sigmoid,
                                           kernel_initializer=INITIALIZER,
                                           kernel_regularizer=tf.nn.l2_loss)
            deeper_units = tf.layers.dense(units,
                                           tf.shape(units)[-1],
                                           kernel_initializer=INITIALIZER,
                                           kernel_regularizer=tf.nn.l2_loss)

            units = sigmoid_gate * units + (1 - sigmoid_gate) * deeper_units
            units = tf.nn.relu(units)

    return units





def expand_tile(units, axis):
    """
    """
    assert axis in (1,2)
    n_time_steps = tf.shape(units)[1]
    repetitions = [1,1,1,1]
    repetitions[axis] = n_time_steps
    return tf.tile(tf.expand_dims(units, axis), repetions)

def additive_self_attention(units, n_hidden=None, n_output_features=None, activation=None):
    """
    return [batch_size, time_steps, n_output_features]
    """
    n_input_features = units.get_shape().as_list()[2]

    if n_hidden is None:
        n_hidden = n_input_features

    if n_output_features is None:
        n_output_features = n_input_features

    units_pairs = tf.concat([expand_tile(units,1), expand_tile(units,2)],3)
    query = tf.layers.dense(units_pairs, n_hidden, activation=tf.tanh, kernel_initializer=INITIALIZER())
    attention = tf.nn.softmax(tf.layers.dense(query,1), dim=2)
    attended_units = tf.reduce_sum(attention * expand_tile(units,1), axis=2)
    output = tf.layers.dense(attended_units, n_output_features, activation, kernel_initializer=INITIALIZER())

    return output

def multiplicative_self_attention(units, n_hidden=None, n_output_features=None, activation=None):
    """

    """
    n_input_features = units.get_shape().as_list()[2]
    if n_hidden is None:
        n_hidden = n_input_features
    if n_output_features is None:
        n_output_features = n_input_features

    queries = tf.layers.dense(expand_tile(units,1), n_hidden, kernel_initializer=INITIALIZER())
    keys = tf.layers.dense(expand_tile(units,2), n_hidden, kernel_initializer=INITIALIZER())

    scores = tf.reduce_sum(queries * keys, axis=3, keep_dims=True)
    attention = tf.nn.softmax(scores, dim=2)
    attention_units = tf.reduce_sum(attention * expand_tile(units,1), axis=2)

    output = tf.layers.dense(attended_units, n_output_features, activation, kernel_initializer=INITIALIZER())

    return output




def variational_dropout(units, keep_prob, fixed_mask_dims=(1,)):
    """
    dropout with the same drop mask for all fixed mask dims
    units: tensor [B T F]
    """

    units_shape = tf.shape(units)
    noise_shape = [units_shape[n] for n in range(len(units.shape))]

    for dim in fixed_mask_dims:
        noise_shape[dim] = 1

    return tf.nn.dropout(units, keep_prob, noise_shape)






