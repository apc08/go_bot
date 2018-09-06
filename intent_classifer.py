# -*- coding: utf-8 -*-

"""
意图分类
"""
import json
import numpy as np

from itertools import chain
from pathlib import Path
from templates import *


import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

#INITIALIZER = tf.orthogonal_initializer
INITIALIZER = xavier_initializer


class TFModel():

    def __init__(self, *args, **kwargs):

        if not hasattr(self, 'sess'):
            raise RuntimeError('Your tensorflow model {} must'
                               'has sess attribute!'.format(self.__class__.__name__))

    def load(self,exclude_scopes=['Optimizer']):
        path = str(self.load_path.resolve())
        if tf.train.checkpoint_exists(path):
            var_list = self._get_saveable_variables(exclude_scopes)
            saver = tf.train.Saver(var_list)
            saver.restore(self.sess, path)

    def save(self,exclude_scopes=['Optimizer']):
        path = str(self.save_path.resolve())
        var_lst = self._get_saveable_variables(exclude_scopes)
        saver = tf.train.Saver(var_list)
        saver.save(self.sess, path)

    def _get_saveable_variables(self, exclude_scopes=[]):
        #all_vars = variables._all_saveable_objects()
        #vars_to_train = [var for var in all_vars if all(sc not in var.name for sc in exclude_scopes)]
        vars_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'model')
        return vars_to_train

    def _get_trainable_variables(self, exclude_scopes=[]):
        #all_vars = tf.global_variables()
        #vars_to_train = [var for var in all_vars if all(sc not in var.name for sc in exclude_scopes)]
        vars_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'model')
        return vars_to_train

    def get_train_op(self,
                     loss,
                     learning_rate,
                     optimizer=None,
                     clip_norm=None,
                     learningable_scopes=None,
                     optimizer_scope_name=None):
        if optimizer_scope_name is None:
            opt_scope = tf.variable_scope("Optimizer")
        else:
            opt_scope = tf.variable_scope("optimizer_scope_name")

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



    def get_train_op_old(self,
                     loss,
                     learning_rate,
                     optimizer=None,
                     clip_norm=None,
                     learnable_scopes=None,
                     optimizer_scope_name=None):
        """ """
        if optimizer_scope_name is None:
            opt_scope = tf.variable_scope('Optimizer')
        else:
            opt_scope = tf.variable_scope(optimizer_scope_name)

        with opt_scope:
            if learnable_scopes is None:
                variables_to_train = tf.global_variables()
            else:
                variables_to_train = []
                for scope_name in learnable_scopes:
                    for var in tf.global_variables():
                        if scope_name in var.name:
                            variables_to_train.append(var)
            if optimizer is None:
                optimizer = tf.train.AdamOptimizer

            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(extra_update_ops):
                opt = optimizer(learning_rate)
                #print(variables_to_train)
                grads_and_vars = opt.compute_gradients(loss, var_list=variables_to_train)

                #for ele in grads_and_vars:
                #    print(ele)

                if clip_norm is not None:
                    grads_and_vars = [(tf.clip_by_norm(grad,clip_norm),var)
                                      for grad, var in grads_and_vars[3:] if grad is not None]
                train_op = opt.apply_gradients(grads_and_vars)

        return train_op

class IntentClassifier(TFModel):

    def __init__(self, **kwargs):
        """
        """
        self.embeddings_dim = kwargs.get('embeddings_dim',100)
        self.cell_size = kwargs.get('cell_size',200)
        self.keep_prob = kwargs.get('keep_prob',0.8)
        self.learning_rate = kwargs.get('learning_rate',3e-4)
        self.grad_clip = kwargs.get('grad_clip',5.0)
        self.vocab_size = kwargs.get('vocab_size', 11595)
        self.teacher_forcing_rate = kwargs.get('teacher_forcing_rate',0.0)
        self.use_attention = kwargs.get('use_attention',False)

        self.num_classes = kwargs.get('num_classes',10)
        self.batch_size  = kwargs.get('batch_size', 32)
        self.sequence_length = kwargs.get("sequence_length",20)
        self.decay_rate = kwargs.get("decay_rate",0.9)
        #self.learning_rate_decay_half_op = tf.assign(self.learning_rate,self.learning_rate * self.decay_rate)
        self.num_filters = kwargs.get("num_filters",3)
        self.hidden_size = kwargs.get('hidden_size',100)
        self.filter_sizes = [3,4,5]
        self.num_filters_total = self.num_filters * len(self.filter_sizes) # how many filters totally
        self.initializer = INITIALIZER()

        self.load_path = Path(kwargs.get('load_path',"")) if kwargs.get('load_path',"") !="" else None
        self.save_path = Path(kwargs.get('save_path',"")) if kwargs.get("save_path","") !="" else None

        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)

        self.loss = self.init_graph()

        self.train_op = self.get_train_op(self.loss, self.learning_rate,
                                          optimizer=tf.train.AdamOptimizer,
                                          clip_norm=self.grad_clip)

        self.sess.run(tf.global_variables_initializer())

        if self.load_path:
            print("加载预训练的model")
            self.load()

        super().__init__(**kwargs)

    def init_placeholders(self):
        self.input_x = tf.placeholder(tf.int32, [None, None],name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")    # y [None, num_classes]
        self.input_y_multilabel = tf.placeholder(tf.float32, [None,self.num_classes], name='input_y_multilabel')
        self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')

        self.global_step = tf.Variable(0,trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        #self.lr_ph = tf.placeholder(tf.float32, [], name='learning_rate')
        #self.decay_steps, self.decay_rate = decay_steps, decay_rate




    def _build_feed_dict(self, x, y=None):
        feed_dict = {
            self.input_x: x,
            self.dropout_keep_prob: 1.0
        }

        if y is not None:
            feed_dict.update({
                self.input_y_multilabel: y,
                #self.lr_ph: self.learning_rate,
                #self.keep_prob_ph: self.keep_prob
                self.dropout_keep_prob:self.keep_prob
            })

        return feed_dict

    def init_graph(self):
        self.init_placeholders()
        loss,multi_label = self.rnn_fn()

        return loss

    def cnn_fn(self,):
        with tf.variable_scope('model') as vs:
            self.embeddings = tf.Variable(tf.random_uniform((self.vocab_size, self.embeddings_dim),-0.1,0.1,
                                      name='embeddings'), dtype=tf.float32)

            self.W_projection = tf.get_variable('W_projection', shape=[self.num_filters_total, self.num_classes], initializer=self.initializer)
            self.b_projection = tf.get_variable('b_projection', shape=[self.num_classes])

            self.embedded_words = tf.nn.embedding_lookup(self.embeddings, self.input_x)  # [None, sentence_length, embed_size]
            self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words,-1)   # [None, sentence_length, embed_size,1]

            # loop each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.variable_scope('conv_pooling-%s' % filter_size):
                    f_map = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embeddings_dim,1,self.num_filters],
                                             initializer=self.initializer)
                    #shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                    conv = tf.nn.conv2d(self.sentence_embeddings_expanded, f_map, strides=[1,1,1,1], padding='VALID', name='conv')
                    b = tf.get_variable('b-%s' % filter_size, [self.num_filters])
                    #shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                    h = tf.nn.relu(tf.nn.bias_add(conv,b), "relu")

                    # [batch_size, 1, 1, num_filters]
                    sequence_length = tf.shape(h)[1]
                    pooled = tf.nn.max_pool(h, ksize=[1, sequence_length-filter_size+1,1,1], strides=[1,1,1,1],padding='VALID',name='pool')
                    pooled_outputs.append(pooled)
            self.h_pool = tf.concat(pooled_outputs,3)#shape:[batch_size, 1, 1, num_filters_total]
            self.h_pool_flat = tf.reshape(self.h_pool,[-1,self.num_filters_total])

            # add dropout
            with tf.variable_scope('dropout'):
                # [None, num_filters_total]
                self.h_drop = tf.nn.dropout(self.h_pool_flat,keep_prob=self.dropout_keep_prob)

            with tf.variable_scope('output'):
                self.logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection

        self.prediction = tf.argmax(self.logits,1,name="predictions")

        #losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
        #loss = tf.reduce_mean(losses)
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * 0.0001
        #loss = loss + l2_losses

        multilabel_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits)
        multilabel_loss = tf.reduce_mean(tf.reduce_sum(multilabel_losses, axis=1))
        multilabel_loss = multilabel_loss + l2_losses

        loss = multilabel_loss

        #loss = multilabel_loss if self.multi_label else multilabel_loss

        return loss, multilabel_loss


    def rnn_fn(self):
        with tf.variable_scope('model'):
            self.embeddings = tf.Variable(tf.random_uniform((self.vocab_size, self.embeddings_dim),-0.1,0.1,
                                      name='embeddings'), dtype=tf.float32)
            self.W_projection = tf.get_variable('W_projection', shape=[self.hidden_size * 2, self.num_classes], initializer=self.initializer)
            self.b_projection = tf.get_variable('b_projection', shape=[self.num_classes])

            self.embedded_words = tf.nn.embedding_lookup(self.embeddings,self.input_x)

            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)

            if self.dropout_keep_prob is not None:
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)

            # [batch_size,sequence_length,hidden_size]
            outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, self.embedded_words, dtype=tf.float32)

            output_rnn = tf.concat(outputs, axis=2) # [batch_size,sequence_length,hidden_size*2]
            self.output_rnn_last = tf.reduce_mean(output_rnn, axis=1) #[batch_size,hidden_size*2]

            self.logits = tf.matmul(self.output_rnn_last, self.W_projection) + self.b_projection #[batch_size, num_classes]

        #losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
        #loss = tf.reduce_mean(losses)
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * 0.00001
        #loss = loss + l2_losses

        multilabel_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits)
        multilabel_loss = tf.reduce_mean(tf.reduce_sum(multilabel_losses, axis=1))
        multilabel_loss = multilabel_loss + l2_losses

        nce_loss = loss = multilabel_loss
        #labels = tf.expand_dims(self.input_y,1) #[batch_size] -> [batch_size,1]
        #nce_loss = tf.reduce_mean(
        #    tf.nn.nce_loss(weights=tf.transpose(self.W_projection),
        #                   biases=self.b_projection,
        #                   labels=labels,
        #                   inputs=self.output_rnn_last,
        #                   num_sampled=self.num_sampled,
        #                   num_classes=self.num_classes,
        #                   partition_strategy='div'))
        #nce_loss = nce_loss + l2_losses

        return loss,nce_loss





    def train_on_batch(self, x, y):
        feed_dict = self._build_feed_dict(x, y)
        loss,_ = self.sess.run([self.loss, self.train_op],
                               feed_dict=feed_dict)
        return loss

    def __call__(self, x):
        feed_dict = self._build_feed_dict(x)
        #y_pred = self.sess.run(self.y_pred_tokens, feed_dict=feed_dict)
        y_pred = self.sess.run(self.logits, feed_dict=feed_dict)

        return y_pred

from data_iterator import *
from dstc_reader import *
from vocab import *
from utils import * #BoWEmbedder
from embedder import *

def main():
    '''
    加载 分类数据
    '''
    dataset_path = './tmp/my_download_of_dstc2'
    dstc_reader = DSTC2DatasetReader()

    data = dstc_reader.read(data_path=dataset_path,
                            dialogs=False)

    intentInterator = Dstc2IntentsDatasetIterator(data)


    # 加载templates
    #template_path = "./tmp/my_download_of_dstc2/dstc2-templates.txt"
    #templates = Templates(ttype=DualTemplate)
    #templates.load(template_path)
    #print(len(templates))
    #print(templates.actions)
    #print(templates.templates)

    # 加载 bow_encoder
    bow_encoder = BoWEmbedder()

    #for x_batch, y_batch in intentInterator.get_batches(batch_size=5):
    #    for x,y in zip(x_batch, y_batch):
    #        print('--' * 50)
    #        print(x)
    #        print(y)


    spcial_tokens = ['<PAD>','<UNK>']
    token_vocab = SimpleVocabulary(spcial_tokens, save_path='tmp/intent_classifier_x.dict')
    #token_vocab = SimpleVocabulary(save_path='tmp/intent_classifier_x.dict')
    label_vocab = SimpleVocabulary(save_path='tmp/intent_classifier_labe.dict')

    print(intentInterator.get_instances('train')[1][:10])
    (sentences, labels) = intentInterator.get_instances('train')

    sentences = [sent.split(" ") for sent in sentences]

    #all_tokens = [tokens for tokens, labels in intentInterator.get_instances('train')]
    #all_labels = [labels for tokens, tags in intentInterator.get_instances('train')]

    all_tokens = sentences
    all_labels = labels
    #all_labels = [[act] for act in templates.actions]



    token_vocab.fit(all_tokens)
    label_vocab.fit(all_labels)

    print(token_vocab.len)
    print(label_vocab.len)

    print('---------- vocab:  -------------------')
    for token in token_vocab.keys():
        print(token)
    print('---------- labels: -------------------')
    for label in label_vocab.keys():
        print(label)

    #print('--------------------------------------')
    #print(token_vocab[1])
    #print('--------------------------------------')
    #print(label_vocab[2])


    # 初始化 intent classifer model

    intent_classifier = IntentClassifier(vocab_size=token_vocab.len,
                                         num_classes=label_vocab.len)

    count = 0
    for epoch in range(10):
        for x_batch, y_batch in intentInterator.get_batches(batch_size=5):
            #for x,y in zip(x_batch, y_batch):
            #    print('--' * 50)
            #    print(x)
            #    print(y)
            x_batch = [x.split(" ") for x in x_batch]
            #print("sent {}".format(x_batch[0]))
            x_batch = np.array(zero_pad(token_vocab(x_batch)))
            #print(label_vocab(y_batch))
            y_batch = np.array(bow_encoder(y_batch, label_vocab))

            #print(x_batch)
            #print(y_batch)

            loss = intent_classifier.train_on_batch(x_batch,y_batch)
            count += 1
            print("epoch %d batch %d loss: %f" % (epoch, count, loss))
            #break

            if count % 100 == 0:
                print("----------- eval ------------------")
                print("pred")
                print(np.asarray(intent_classifier(x_batch)>0).astype(np.int32))
                print("label")
                print(y_batch)

    #print(templates.actions)
    #print(labels)


if __name__ == '__main__':
    main()




