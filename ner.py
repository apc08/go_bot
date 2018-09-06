# -*- coding: utf-8 -*-

"""
测试ner
"""

import deeppavlov
from deeppavlov.core.data.utils import download_decompress

#download_decompress('http://files.deeppavlov.ai/deeppavlov_data/conll2003_v2.tar.gz', 'data/')

from deeppavlov.dataset_readers.conll2003_reader import Conll2003DatasetReader
dataset = Conll2003DatasetReader().read('data/')

for sample in dataset['train'][:4]:
    for token,tag in zip(*sample):
        print('%s\t%s' % (token,tag))
    print()

# 初始化词典
from deeppavlov.core.data.simple_vocab import SimpleVocabulary

special_tokens = ['<UNK>']

token_vocab = SimpleVocabulary(special_tokens, save_path='model/ner/token.dict')
tag_vocab = SimpleVocabulary(save_path='model/ner/tag.dict')

all_tokens_by_sentences = [tokens for tokens, tags in dataset['train']]
all_tags_by_sentences = [tags for tokens, tags in dataset['train']]

token_vocab.fit(all_tokens_by_sentences)
tag_vocab.fit(all_tags_by_sentences)

print(token_vocab([['How', 'to', 'do', 'a', 'barrel', 'roll', '?']]))
print(tag_vocab([['O', 'O', 'O'], ['B-ORG', 'I-ORG']]))

import numpy as np

print(token_vocab([np.random.randint(0,512,size=10)]))


from deeppavlov.core.data.data_learning_iterator import DataLearningIterator

data_iterator = DataLearningIterator(dataset)

print("batch 数据格式:")
print(next(data_iterator.gen_batches(5, shuffle=True)))

from deeppavlov.models.preprocessors.mask import Mask
get_mask = Mask()

print(get_mask([['Try', 'to', 'get', 'the', 'mask'], ['Check', 'paddings']]))

import tensorflow as tf
import numpy as np

np.random.seed(42)
tf.set_random_seed(42)

def get_embeddings(indices, vocabulary_size, emb_dim):
    emb_mat = np.random.randn(vocabulary_size, emb_dim).astype(np.float32) / np.sqrt(emb_dim)

    emb_mat = tf.Variable(emb_mat, name='Embeddings', trainable=True)
    emb = tf.nn.embedding_lookup(emb_mat, indices)
    return emb


x = tf.random_normal(shape=[2,10,100])
y = tf.layers.conv1d(x,filters=200, kernel_size=8)
print(y)

y_with_padding = tf.layers.conv1d(x, filters=200, kernel_size=0, padding='same')
print(y_with_padding)


def conv_net(units, n_hidden_list, cnn_filter_width, activation=tf.nn.relu):
    for n_hidden in n_hidden_list:
        units = tf.layers.conv1d(units,n_hidden,
                                 cnn_filter_width,
                                 padding='same')
        units = activation(units)

    return units

l = tf.random_normal([1,4,3])  # shape [batch_size, number_of_tokens, num of class]
indices = tf.placeholder(tf.int32,[1,4])

#
p = tf.one_hot(indices, depth=3)
loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=p, logits=l)

print(loss_tensor)

mask = tf.placeholder(tf.float32, shape=[1,4])
loss_tensor *= mask

loss = tf.reduce_mean(loss_tensor)

def masked_cross_entropy(logits, label_indices, number_of_tags, mask):
    ground_truth_labels = tf.one_hot(label_indices, depth=number_of_tags)
    loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=ground_truth_labels,logits=logits)

    loss_tensor *= mask
    loss = tf.reduce_mean(loss_tensor)

    return loss


class NerNetwork:

    def __init__(self,
                 n_tokens,
                 n_tags,
                 token_emb_dim=100,
                 n_hidden_list=(128,),
                 cnn_filter_width = 7,
                 use_batch_norm=False,
                 embeddings_dropout=False,
                 top_dropout=False,
                 **kwargs):
        # --------------------------------------------
        self.learning_rate_ph = tf.placeholder(tf.float32,[])
        self.dropout_keep_ph = tf.placeholder(tf.float32,[])
        self.token_ph = tf.placeholder(tf.int32,[None,None], name='token_ind_ph')
        self.mask_ph =tf.placeholder(tf.float32, [None,None], name='Mask_ph')
        self.y_ph = tf.placeholder(tf.int32, [None, None], name='y_ph')

        # ----------- building the network -----------
        emb = get_embeddings(self.token_ph, n_tokens, token_emb_dim)
        # --------------------------------------------
        emb = tf.nn.dropout(emb, self.dropout_keep_ph, (tf.shape(emb)[0],1,tf.shape(emb)[2]))

        units = conv_net(emb,n_hidden_list, cnn_filter_width)

        units = tf.nn.dropout(units, self.dropout_keep_ph,
                              (tf.shape(units)[0],1,tf.shape(units)[2]))

        logits = tf.layers.dense(units, n_tags, activation=None)

        print("输出 shape:")
        print(logits)

        self.predictions = tf.argmax(logits, 2)

        print("label shape:")
        print(self.y_ph)

        # ------------ loss and train ops ------------
        self.loss = masked_cross_entropy(logits, self.y_ph, n_tags, self.mask_ph)

        optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
        self.train_op = optimizer.minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def __call__(self, tok_batch, mask_batch):
        feed_dict = {self.token_ph: tok_batch,
                     self.mask_ph: mask_batch,
                     self.dropout_keep_ph: 1.0}
        return self.sess.run(self.predictions, feed_dict)

    def train_on_batch(self, tok_batch, tag_batch, mask_batch, dropout_keep_prob,learning_rate):
        feed_dict = {self.token_ph: tok_batch,
                     self.y_ph: tag_batch,
                     self.mask_ph: mask_batch,
                     self.dropout_keep_ph: dropout_keep_prob,
                     self.learning_rate_ph: learning_rate}

        self.sess.run(self.train_op, feed_dict)

nernet = NerNetwork(len(token_vocab),
                        len(tag_vocab),
                        n_hidden_list=[100,100])

from deeppavlov.models.ner.evaluation import precision_recall_f1
from deeppavlov.core.data.utils import zero_pad
from itertools import chain


def eval_valid(network, batch_generator):
    total_true = []
    total_pred = []

    for x, y_true in batch_generator:

        print(x[0])
        print(y_true[0])
        x_inds = token_vocab(x)

        # pad the indices batch with zeros
        x_batch = zero_pad(x_inds)

        mask = get_mask(x)

        y_inds = network(x_batch,mask)

        y_inds = [y_inds[n][:len(x[n])] for n,y in enumerate(y_inds)]
        y_pred = tag_vocab(y_inds)

        #
        total_true.extend(chain(*y_true))
        total_pred.extend(chain(*y_pred))

    res = precision_recall_f1(total_true, total_pred, print_results=True)



batch_size = 16
n_epochs = 20
learning_rate = 0.001
dropout_keep_prob = 0.5


for epoch in range(n_epochs):

    for x,y in data_iterator.gen_batches(batch_size,'train'):
        x_inds = token_vocab(x)
        y_inds = tag_vocab(y)

        x_batch = zero_pad(x_inds)
        y_batch = zero_pad(y_inds)

        mask = get_mask(x)
        nernet.train_on_batch(x_batch, y_batch, mask, dropout_keep_prob, learning_rate)
    print("Evaluating the model on valid part of the dataset")
    eval_valid(nernet, data_iterator.gen_batches(batch_size, 'valid'))






