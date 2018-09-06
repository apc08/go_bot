# -*- coding: utf-8 -*-

'''
ner model
'''

from dstc_reader import *
from vocab import *
from data_iterator import *

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np

from tf_layers import embedding_layer, character_embedding_network, variational_dropout
from tf_layers import bi_rnn, stacked_cnn, INITIALIZER

np.random.seed(42)
tf.set_random_seed(42)

#INITIALIZER = tf.orthogonal_initializer
#INITIALIZER = xavier_initializer

class TFModel():

    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'sess'):
            raise RuntimeError('Your tensorflow model {} must '
                               'has sess attribute!'.format(self.__class__.__name__))

    def load(self, exclude_scopes=['Optimizer']):
        path = str(self.load_path.resolve())
        if tf.train.checkpoint_exists(path):
            var_list = self._get_saveable_variables(exclude_scopes)
            #print(var_list)
            saver = tf.train.Saver(var_list)
            saver.restore(self.sess, path)

    def save(self, exclude_scopes=['Optimizer']):
        path = str(self.save_path.resolve())
        var_list = self._get_saveable_variables(exclude_scopes)
        #print(var_list)
        saver = tf.train.Saver(var_list)
        saver.save(self.sess, path)

    def _get_saveable_variables(self, exclude_scopes=[]):
        vars_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
        return vars_to_train

    def _get_trainable_variables(self, exclude_scopes=[]):
        vars_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
        #print(vars_to_train)
        return vars_to_train

    def get_train_op(self,
                     loss,
                     learning_rate,
                     optimizer=None,
                     clip_norm=None,
                     learningable_scopes=None,
                     optimizer_scope_name=None):
        if optimizer_scope_name is None:
            opt_scope = tf.variable_scope('Optimizer')
        else:
            opt_scope = tf.variable_scope(optimizer_scope_name)

        with opt_scope:
            if learningable_scopes is None:
                variables_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=learningable_scopes)
            else:
                variables_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')


            if optimizer is None:
                optimizer = tf.train.AdamOptimizer

            opt = optimizer(learning_rate)

            grads_and_vars = opt.compute_gradients(loss, var_list=variables_to_train)

            if clip_norm is not None:
                grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var) for (grad, var) in grads_and_vars]

            train_op = opt.apply_gradients(grads_and_vars)

        return train_op


class NerNetowrk(TFModel):
    """
    """
    GRAPH_PARAMS = ["n_tags",
                    "char_emb_dim",
                    "capitalization_dim",
                    "additional_features",
                    "use_char_embeddings",
                    "additional_features",
                    "net_type",
                    "cell_type",
                    "char_filter_width",
                    "cell_type"]

    def __init__(self,
                 n_tags,   # features dimensions
                 token_emb_dim = None,
                 char_emb_dim = None,
                 capitalization_dim = None,
                 pos_features_dim = None,
                 additional_features = None,
                 net_type='rnn',
                 cell_type='lstm',
                 use_cudnn_rnn=False,
                 two_dense_on_top=False,
                 n_hidden_list=(128,),
                 cnn_filter_width = 7,
                 use_crf = False,
                 token_emb_mat = None,
                 char_emb_mat = None,
                 use_batch_norm=False,
                 dropout_keep_prob=0.5,
                 embeddings_dropout=False,
                 top_dropout = False,
                 intra_layer_dropout=False,
                 l2_reg = 0.0,
                 clip_grad_norm=5.0,
                 learning_rate = 3e-3,
                 gpu = None,
                 seed=None,
                 lr_drop_patience=5,
                 lr_drop_value=0.1,
                 **kwarags):
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self._learning_rate = learning_rate
        self._lr_drop_patience = lr_drop_patience
        self._lr_drop_value = lr_drop_value
        self._add_training_placeholders(dropout_keep_prob, learning_rate)
        self._xs_ph_list = []
        self._y_ph = tf.placeholder(tf.int32, [None, None], name='y_ph')
        self._input_features = []


        # ============ Building input featues ===========================

        # Token embeddings
        self._add_word_embeddings(token_emb_mat, token_emb_dim)

        # Masks for different lengths utterances
        self.mask_ph = self._add_mask()

        # Char embeddings using highway CNN wiith max pooling
        if char_emb_mat is not None and char_emb_dim is not None:
            self._add_char_embeddings(char_emb_mat)

        # capitalization features
        if capitalization_dim is not None:
            self._add_capitalization(capitalization_dim)

        # part of speech featues
        if pos_features_dim is not None:
            self._add_pos(pos_features_dim)

        # anything you want
        if additional_features is not None:
            self._add_additional_features(additional_features)

        features = tf.concat(self._input_features, axis=2)
        if embeddings_dropout:
            features = variational_dropout(features, self._cropout_ph)

        # =============== building the network ==========================

        if net_type == 'rnn':
            units = self._build_rnn(features, n_hidden_list, cell_type, intra_layer_dropout, self.mask_ph)
        elif net_type == 'cnn':
            units = self._build_cnn(features, n_hidden_list, cnn_filter_width, use_batch_norm)

        self._logits = self._build_top(units, n_tags, n_hidden_list[-1], top_dropout, two_dense_on_top)

        self.train_op, self.loss = self._build_train_predict(self._logits, self.mask_ph, n_tags,
                                                             use_crf, clip_grad_norm, l2_reg)

        self.predict = self.predict_crf if use_crf else self.predict_no_crf


        # =============== initialization the session =====================

        sess_config = tf.ConfigProto(allow_soft_replacement=True)
        sess_config.gpu_options.allow_growth = True
        if gpu is not None:
            sess_config.gpu_options.visible_device_list = str(gpu)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        super().__init__(**kwargs)
        self.load()


    def _add_training_placeholders(self, dropout_keep_prob, learning_rate):
        self.learning_rate_ph = tf.placeholder_with_default(learning_rate, shape=[], name='learning_rate')
        self._dropout_ph = tf.placeholder_with_default(dropout_keep_prob, shape=[], name='dropout')
        self.train_ph = tf.placeholder_with_default(False, shape=[], name='is_training')

    def _add_word_embeddings(self, token_emb_mat, token_emb_dim=None):
        if token_emb_mat is None:
            token_ph = tf.placeholder(tf.float32, [None, None, token_emb_dim], name='Token_Ind_ph')
            emb = token_ph
        else:
            token_ph = tf.placeholder(tf.int32, [None,None], name='Tokne_Ind_ph')
            emb = embedding_layer(token_ph, token_emb_mat)
        self._xs_ph_list.append(token_ph)
        self._input_features.append(emb)


    def _add_mask(self):
        mask_ph = tf.placeholder(tf.float32, [None, None], name='Mask_ph')
        self._xs_ph_list.append(mask_ph)
        return mask_ph

    def _add_char_embeddings(self, char_emb_mat):
        character_indices_ph = tf.placeholder(tf.int32, [None, None, None], name='Char_ph')
        char_embs = character_embedding_network(character_indices_ph, emb_mat=char_emb_mat)
        self._xs_ph_list.append(character_indices_ph)
        self._input_features.append(char_embs)

    def _add_pos(self, pos_features_dim):
        pos_ph = tf.placeholder(tf.float32, [None, None, pos_features_dim], name='POS_ph')
        self._xs_ph_list.append(pos_ph)
        self._input_features.append(pos_ph)

    def _add_attional_features(self,feature_list):
        for feature, dim in features_list:
            feat_ph = tf.placeholder(tf.float32, [None, None, dim], name=feature+'_ph')
            self._x_ph_list.append(feat_ph)
            self._input_features.append(feat_ph)

    def _build_rnn(self, units, n_hidden_list, cell_type, intra_layer_dropout):
        for n, n_hidden in enumerate(n_hidden_list):
            units, _ = bi_rnn(units, n_hidden, cell_type=cell_type, name='Layer_' + str(n))
            units = tf.concat(units, -1)
            if intra_layer_dropout and n != len(n_hidden_list) - 1:
                units = variational_dropout(units, self._dropout_ph)

        return units

    def _build_cnn(self, units, n_hidden_list, cnn_filter_width, use_batch_norm):
        units = stacked_cnn(units, n_hidden_list, cnn_filter_width, use_batch_norm, training_ph=self.training_ph)
        return units

    def _build_top(self, units, n_tags, n_hidden, top_dropout, two_dense_on_top):
        if top_dropout:
            units = variational_dropout(units, self._dropout_ph)
        if two_dense_on_top:
            units = tf.layers.dense(units, n_hidden, activation=tf.nn.relu,
                                    kernel_initializer=INITIALIZER(),
                                    kernel_regularizer=tf.nn.l2_loss)
        logits = tf.layers.dense(units, n_tags, activation=None,
                                 kernel_initializer=INITIALIZER(),
                                 kernel_regularizer=tf.nn.l2_loss)

        return logits

    def _build_train_predict(self, logits, mask, n_tags, use_crf, clip_grad_norm, l2_reg):
        """
        """
        if use_crf:
            sequence_lengths = tf.reduce_sum(mask, axis=1)
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits, self._y_ph, sequence_lengths)
            loss_tensor = -log_likelihood
            self._tranisition_params = transition_params

        else:
            ground_truth_labels = tf.one_hot(self._y_ph, n_tags)
            loss_tensor = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_labels, logits=logits)
            loss_tensor = loss_tensor * mask
            self._y_pred = tf.argmax(logits, axis=-1)

        loss = tf.reduce_mean(loss_tensor)

        if l2_reg > 0:
            loss += l2_reg * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # optimizer = partial(tf.train.MomentumOptimizer, momentum=0.9, use_nesterov=True)
        optimizer = tf.train.AdamOptimizer
        train_op = self.get_train_op(loss, self.learning_rate_ph, optimizer, clip_norm=clip_grad_norm)

        return train_op, loss

    def predict_no_crf(self, xs):
        feed_dict = self._fill_feed_dict(xs)
        pred_idxs,mask = self.sess.run([self._y_pred, self.mask_ph], feed_dict)

        # filter by sequence length
        sequence_lengths = np.sum(mask, axis=1).astype(np.int32)
        pred = []

        for utt, l in zip(pred_idxs, sequence_lengths):
            pred.append(utt[:l])
        return pred

    def predict_crf(self, xs):
        feed_dict = self._fill_feed_dict(xs)

        logits, trans_params, mask = self.sess.run([self._logits,
                                                    self._transition_params,
                                                    self.mask_ph],
                                                   feed_dict=feed_dict)
        sequence_lengths = np.maximum(np.sum(mask, axis=1).astype(np.int32),1)

        # 使用 维特比 算法进行解码
        y_pred = []
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:int(sequence_length)]   # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            y_pred += [viterbi_seq]

        return y_pred

    def _fill_feed_dict(self, xs, y=None, learning_rate=None, train=False):
        assert len(xs) == len(self._xs_ph_list)
        xs = list(xs)
        xs[0] = np.array(xs[0])
        feed_dict = {ph: x for ph,x in zip(self._xs_ph_list, xs)}
        if y is not None:
            feed_dict[self._y_ph] = y
        if learning_rate is not None:
            feed_dict[self.learning_rate_ph] = learning_rate
        feed_dict[self.training_ph] = train
        if not train:
            feed_dict[self._dropout_ph] = 1.0
        return feed_dict

    def __call__(self, *args, **kwargs):
        if len(args[0]) == 0 or (len(args[0]) == 1 and len(args[0][0]) == 0):
            return []
        return self.predict(args)

    def train_on_batch(self, *args):
        *xs,y=args
        feed_dict = self._fill_feed_dict(xs, y, train=True, learning_rate=self._learning_rate)
        _,loss = self.sess.run([self.train_op,self.loss], feed_dict)
        return loss

    def process_event(self, event_name, data):
        pass


class Mask():
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self,tokens_batch, **kwargs):
        batch_size = len(tokens_batch)
        max_len = max(len(utt) for utt in tokens_batch)
        mask = np.zeros([batch_size, max_len], dtype=np.float32)
        for n, utterance in enumerate(tokens_batch):
            mask[n, :len(utterance)] = 1

        return mask



# ------------------------------------------------------------------------------------------------

class NerSimpleNetwork(TFModel):

    def __init__(self,
                 n_tokens,
                 n_tags,
                 token_emb_dim=100,
                 net_type='rnn',
                 cell_type='lstm',
                 two_dense_on_top=False,
                 n_hidden_list=(128,),
                 cnn_filter_width=7,
                 use_crf=False,
                 token_emb_mat=None,
                 embeddings_dropout=False,
                 l2_reg = 0.0,
                 clip_grad_norm=5.0,
                 learning_rate = 3e-3,
                 dropout_keep_prob=0.5,
                 lr_drop_patience=5,
                 lr_drop_value=0.1,
                 intra_layer_dropout=False,
                 top_dropout=False,
                 use_batch_norm=False,
                 seed=42,
                 gpu=None,
                 load_path=None,
                 save_path=None,
                 **kwargs):
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.n_tokens = n_tokens
        self.n_tags = n_tags

        self._learning_rate = learning_rate
        self._learning_drop_patience = lr_drop_patience
        self._lr_drop_value = lr_drop_value
        self._add_training_placeholders(dropout_keep_prob, learning_rate)
        self._xs_ph_list = []
        self._y_ph = tf.placeholder(tf.int32, [None, None], name='y_ph')
        self._input_features = []

        self.load_path = Path(load_path) if load_path else None
        self.save_path = Path(save_path) if save_path else None

        # -------------------- 构建输入特征 -----------------------------

        with tf.variable_scope("model"):
            # token embeddings
            #def _add_word_embeddings(self, token_emb_mat, n_tokens=None, token_emb_dim=None):
            self._add_word_embeddings(token_emb_mat,n_tokens ,token_emb_dim)

            # Mask
            self.mask_ph = self._add_mask()

            features = tf.concat(self._input_features, axis=2)

            if embeddings_dropout:
                features = variational_dropout(features, self._dropout_ph)

        # -------------------- 构建 网络 --------------------------------

        #with tf.variable_scope("model"):
            if net_type == 'rnn':
                units = self._build_rnn(features, n_hidden_list, cell_type, intra_layer_dropout, self.mask_ph)

            elif net_type == 'cnn':
                units = self._build_cnn(features, n_hidden_list, cnn_filter_width, use_batch_norm)

            self._logits = self._build_top(units, n_tags, n_hidden_list[-1], top_dropout, two_dense_on_top)

            self.train_op, self.loss = self._build_train_predict(self._logits, self.mask_ph, n_tags,
                                                                 use_crf, clip_grad_norm, l2_reg)
            self.predict = self.predict_crf if use_crf else self.predict_no_crf

        # -------------------- intiialize the sess ----------------------

        #sess_config = tf.ConfigProto(allow_soft_replacement=True)
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True

        if gpu is not None:
            sess_config.gpu_options.visible_device_list = str(gpu)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        super().__init__(**kwargs)
        if self.load_path:
            print("加载预训练的model")
            self.load()

    def _add_training_placeholders(self, dropout_keep_prob, learning_rate):
        self.learning_rate_ph = tf.placeholder_with_default(learning_rate, shape=[], name='learning_rate')
        self._dropout_ph = tf.placeholder_with_default(dropout_keep_prob, shape=[], name='dropout')
        self.training_ph = tf.placeholder_with_default(False, shape=[], name='is_training')

    def _add_word_embeddings(self, token_emb_mat, n_tokens=None, token_emb_dim=None):
        if token_emb_mat is not None:
            token_ph = tf.placeholder(tf.float32, [None, None, token_emb_dim], name='Token_Ind_ph')
            emb = token_ph
        else:
            token_ph = tf.placeholder(tf.int32, [None, None], name="Token_Ind_ph")
            emb = embedding_layer(token_ph, token_emb_mat, n_tokens=self.n_tokens,
                                  token_embedding_dim=token_emb_dim)

        self._xs_ph_list.append(token_ph)
        self._input_features.append(emb)

    def _add_mask(self):
        mask_ph = tf.placeholder(tf.float32, [None, None], name='Mask_ph')
        self._xs_ph_list.append(mask_ph)
        return mask_ph

    def _build_rnn(self, units, n_hidden_list, cell_type, intra_layer_dropout, mask):
        for n, n_hidden in enumerate(n_hidden_list):
            units, _ = bi_rnn(units, n_hidden, cell_type=cell_type, name='Layer_' + str(n))
            units = tf.concat(units, -1)

            if intra_layer_dropout and n != len(n_hidden_list) - 1:
                units = variational_dropout(units, self._dropout_ph)

        return units

    def _build_cnn(self, units, n_hidden_list, cnn_filter_width, use_batch_norm):
        """
        """
        units = stacked_cnn(units, n_hidden_list, cnn_filter_width, use_batch_norm, training_ph=self.training_ph)
        return units

    def _build_top(self, units, n_tags, n_hidden, top_dropout, two_dense_on_top):
        if top_dropout:
            units = variational_dropout(units, self._dropout_ph)

        if two_dense_on_top:
            units = tf.layers.dense(units, n_hidden, activation=tf.nn.relu,
                                    kernel_initializer=INITIALIZER(),
                                    kernel_regularizer=tf.nn.l2_loss)
        logits = tf.layers.dense(units, n_tags, activation=None,
                                 kernel_initializer=INITIALIZER(),
                                 kernel_regularizer=tf.nn.l2_loss)

        return logits

    def _build_train_predict(self, logits, mask, n_tags, use_crf, clip_grad_norm, l2_reg):
        """
        """
        if use_crf:
            sequence_lengths = tf.reduce_sum(mask, axis=1)
            log_likelihood , transition_params = tf.contrib.crf.crf_log_likelihood(logits, self._y_ph, sequence_lengths)
            loss_tensor = - log_likelihood
            self._transition_params = transition_params
        else:
            ground_truth_labels = tf.one_hot(self._y_ph, n_tags)
            loss_tensor = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_labels, logits=logits)
            loss_tensor = loss_tensor * mask
            self._y_pred = tf.argmax(logits, axis=-1)

        loss = tf.reduce_mean(loss_tensor)

        # l2 regularization
        if l2_reg > 0:
            loss += l2_reg * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        #
        optimizer = tf.train.AdamOptimizer
        train_op = self.get_train_op(loss, self.learning_rate_ph, optimizer, clip_norm=clip_grad_norm)

        return train_op, loss

    def predict_no_crf(self, xs):
        feed_dict = self._fill_feed_dict(xs)
        pred_idxs, mask = self.sess.run([self._y_pred, self.mask_ph], feed_dict)

        # Filter  by sequence length
        sequence_lengths = np.sum(mask, axis=1).astype(np.int32)
        pred = []

        for utt, l in zip(pred_idxs, sequence_lengths):
            pred.append(utt[:l])

        return pred

    def predict_crf(self, xs):
        feed_dict = self._fill_feed_dict(xs)
        logits, trans_params, mask = self.sess.run([self._logits,self._transition_params,self.mask_ph],
                                                   feed_dict=feed_dict)
        sequence_lengths = np.maximum(np.sum(mask, axis=1).astype(np.int32),1)

        y_pred = []
        for logit, sequence_length  in zip(logits, sequence_lengths):
            logit = logit[:int(sequence_length)]
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            y_pred += [viterbi_seq]

        return y_pred

    def _fill_feed_dict(self, xs, y=None, learning_rate=None, train=False):
        assert len(xs) == len(self._xs_ph_list)
        xs = list(xs)
        xs[0] = np.array(xs[0])

        feed_dict = {ph: x for ph, x in zip(self._xs_ph_list, xs)}

        if y is not None:
            feed_dict[self._y_ph] = y

        if learning_rate is not None:
            feed_dict[self.learning_rate_ph] = learning_rate
        feed_dict[self.training_ph] = train
        if not train:
            feed_dict[self._dropout_ph] = 1.0
        return feed_dict

    def __call__(self, *args, **kwargs):
        if len(args[0]) == 0 or (len(args[0]) == 1 and len(args[0][0]) == 0):
            return []
        return self.predict(args)

    def train_on_batch(self, *args):
        xs, y = args[:-1],args[-1]
        feed_dict = self._fill_feed_dict(xs,y,train=True,learning_rate=self._learning_rate)
        _,loss = self.sess.run([self.train_op, self.loss], feed_dict)

        return loss



import deeppavlov
from deeppavlov.dataset_readers.conll2003_reader import Conll2003DatasetReader
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.models.ner.evaluation import precision_recall_f1
from utils import *
from itertools import chain


def main():
    """
    """
    dataset = Conll2003DatasetReader().read('data/')

    #for sample in dataset['train'][:4]:
    #    for token, tag in zip(*sample):
    #        print('%s\t%s' % (token, tag))
    #    print()

    special_tokens = ['<UNK>']
    token_vocab = SimpleVocabulary(special_tokens, save_path='model/ner/token.dict')
    tag_vocab   = SimpleVocabulary(save_path='model/ner/tag.dict')

    all_tokens_by_sentences = [tokens for tokens, tags in dataset['train']]
    all_tags_by_sentences = [tags for tokens, tags in dataset['train']]

    token_vocab.fit(all_tokens_by_sentences)
    tag_vocab.fit(all_tags_by_sentences)

    data_iterator = DataLearningIterator(dataset)

    print("batch 数据格式: ")
    print(next(data_iterator.gen_batches(5, shuffle=True)))

    get_mask = Mask()

    print(get_mask([['Try','to','get','the','mask'], ['Check','paddings']]))


    def eval_valid(network, batch_generator):
        total_true = []
        total_pred = []

        for x, y_true in batch_generator:
            x_inds = token_vocab(x)
            x_batch = zero_pad(x_inds)
            mask = get_mask(x)

            y_inds = network(x_batch,mask)

            y_inds = [y_inds[n][:len(x[n])] for n,y in enumerate(y_inds)]
            y_pred = tag_vocab(y_inds)

            total_true.extend(chain(*y_true))
            total_pred.extend(chain(*y_pred))

        print(y_pred)
        print(y_true)


        res = precision_recall_f1(total_true,total_pred,print_results=True)

    # 测试model

    batch_size=5
    n_epochs = 20
    learning_rate = 0.000005
    dropout_keep_prob = 0.5
    model_checkpoint = "./model/ner/ner_model.ckpt"



    nernet = NerSimpleNetwork(len(token_vocab),
                              len(tag_vocab),
                              n_hidden_list=[100,100],
                              use_crf=True,
                              save_path=model_checkpoint,
                              load_path=model_checkpoint,
                              #load_path=None,
                              )


    print("Evaluating the model on valid part of the dataset")
    eval_valid(nernet, data_iterator.gen_batches(batch_size,'valid'))
    count = 0
    for epoch in range(n_epochs):
        for x,y in data_iterator.gen_batches(batch_size, 'train'):
            x_inds = token_vocab(x)
            y_inds = tag_vocab(y)

            x_batch = zero_pad(x_inds)
            y_batch = zero_pad(y_inds)

            mask = get_mask(x)

            #nernet.train_on_batch(x_batch, mask, y_batch, dropout_keep_prob,learning_rate)
            loss = nernet.train_on_batch(x_batch, mask, y_batch)
            count += 1

            if count % 50 == 0:
                print("epoch %d loss %f" % (count, loss))

            if count % 1000 == 0:
                print("Evaluating the model on valid part of the dataset")
                eval_valid(nernet, data_iterator.gen_batches(batch_size,'valid'))
            if count % 1000 == 0:
                print("save model")
                nernet.save()










if __name__ == '__main__':
    main()
















