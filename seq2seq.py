# -*- coding: utf-8 -*-

'''
测试 数据加载
'''

import deeppavlov
import json
import numpy as np
import tensorflow as tf

from itertools import chain
from pathlib import Path

from deeppavlov.core.commands.train import build_model_from_config
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress
from deeppavlov.core.common.registry import register


# 加载数据
from deeppavlov.core.data.utils import download_decompress

#download_decompress('http://files.deeppavlov.ai/datasets/personachat_v2.tar.gz', './personachat')

# 等级 数据加载函数
@register('personachat_dataset_reader')
class PersonaChatDatasetReader(DatasetReader):
    '''
    加载chit数据
    '''
    def read(self, dir_path, mode='self_original'):
        dir_path = Path(dir_path)
        dataset = {}

        for dt in ['train', 'valid', 'test']:
            dataset[dt] = self._parse_data(dir_path / '{}_{}.txt'.format(dt, mode))

        return dataset

    @staticmethod
    def _parse_data(filename):
        examples = []
        print(filename)
        curr_persona = []
        curr_dialog_history = []
        persona_done = False
        with filename.open('r') as fin:
            for line in fin:
                line = ' '.join(line.strip().split(' ')[1:])
                your_persona_pref = 'your persona: '
                if line[: len(your_persona_pref)] == your_persona_pref and persona_done:
                    curr_persona = [line[len(your_persona_pref):]]
                    curr_dialog_history = []
                    persona_done = False
                elif line[:len(your_persona_pref)] == your_persona_pref:
                    curr_persona.append(line[len(your_persona_pref):])
                else:
                    persona_done = True
                    x,y,_,candidates = line.split('\t')
                    candidates = candidates.split('|')
                    example = {
                        'persona': curr_persona,
                        'x': x,
                        'y': y,
                        'dialog_history':curr_dialog_history[:],
                        'candidates': candidates,
                        'y_idx': candidates.index(y)
                    }

                    curr_dialog_history.extend([x,y])
                    examples.append(example)

        return examples

data = PersonaChatDatasetReader().read('./personachat')

for k in data:
    print(k, len(data[k]))


print(data['train'][0])


# 定义数据便利类
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator

#@register
class PersonCharChatIterator(DataLearningIterator):

    def split(self,*args, **kwargs):
        for dt in ['train', 'valid', 'test']:
            setattr(self,dt, self._to_tuple(getattr(self,dt)))

    @staticmethod
    def _to_tuple(data):
        """ """
        return list(map(lambda x: (x['x'], x['y']),data))


# 展示数据
iterator = PersonCharChatIterator(data)

batch = [el for el in iterator.gen_batches(5, 'train')][0]

for x,y in zip(*batch):
    print('x: ', x)
    print('y: ', y)
    print('--------------')



# 调用分词
from deeppavlov.models.tokenizers.lazy_tokenizer import LazyTokenizer

tokenizer = LazyTokenizer()
tokenizer(['Hello my friend'])

# 构建vocab
from deeppavlov.core.data.simple_vocab import SimpleVocabulary

@register('dialog_vocab')
class DialogVocab(SimpleVocabulary):

    def fit(self, *args):
        tokens = chain(*args)
        super().fit(tokens)

    def __call__(self, batch, **kwargs):
        indices_batch = []
        for utt in batch:
            tokens = [self[token] for token in utt]
            indices_batch.append(tokens)

        return indices_batch

vocab = DialogVocab(
    save_path='./vocab.dict',
    load_path='./vocab.dict',
    min_freq=2,
    special_tokens = ('<PAD>','<BOS>','<EOS>','<UNK>'),
    unk_token='<UNK>')

vocab.fit(tokenizer(iterator.get_instances(data_type='train')[0]), tokenizer(iterator.get_instances(data_type='train')[1]))

vocab.save()

print(vocab.freqs.most_common(10))

print(len(vocab))

print(vocab([['<BOS>', 'hello', 'my', 'friend', 'there_is_no_such_word_in_dataset', 'and_this', '<EOS>', '<PAD>']]))

# 批次补齐
from deeppavlov.core.models.component import Component

#@register
class SentencePadder(Component):
    def __init__(self, length_limit, pad_token_id=0, start_token_id=1, end_token_id=2, *args, **kwargs):
        self.length_limit = length_limit
        self.pad_token_id = pad_token_id
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id

    def __call__(self, batch):
        for i in range(len(batch)):
            batch[i] = batch[i][:self.length_limit]
            batch[i] = [self.start_token_id] + batch[i] + [self.end_token_id]
            batch[i] += [self.pad_token_id] * (self.length_limit + 2 - len(batch[i]))

        return batch


padder = SentencePadder(length_limit=6)
vocab(padder(vocab([['hello', 'my', 'friend', 'there_is_no_such_word_in_dataset', 'and_this']])))

# 定义编码器
def encoder(inputs, inputs_len, embedding_matrix, cell_size, keep_prob=1.0):
    """
    """
    with tf.variable_scope("encoder"):
        x_emb = tf.nn.dropout(tf.nn.embedding_lookup(embedding_matrix,inputs), keep_prob=keep_prob)

        encoder_cell = tf.nn.rnn_cell.GRUCell(
            num_units=cell_size,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='encoder_cell')

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                           inputs=x_emb,
                                                           sequence_length=inputs_len,
                                                           dtype=tf.float32)

        return encoder_outputs, encoder_state

# 测试编码器运行是否正确
#tf.reset_default_graph()
#vocab_size = 100
#hidden_dim = 100

#inputs = tf.cast(tf.random_uniform(shape=[32,10]) * vocab_size,tf.int32)
#mask = tf.cast(tf.random_uniform(shape=[32,10]) * 2, tf.int32)
#inputs_len = tf.reduce_sum(mask, axis=1)
#embedding_matrix = tf.random_uniform(shape=[vocab_size, hidden_dim])

#encoder(inputs, inputs_len, embedding_matrix, hidden_dim)


# 定义 注意力机制

def softmax_mask(values, mask):
    INF = 1e30
    return -INF * (1 - tf.cast(mask, tf.float32)) + values

def dot_attention(memory, state, mask, scope='dot_attention'):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        logits = tf.matmul(memory, tf.expand_dims(state, axis=1), transpose_b=True)
        logits = tf.squeeze(logits, [2])

        ligits = softmax_mask(logits, mask)

        att_weights = tf.expand_dims(tf.nn.softmax(logits), axis=2)

        att = tf.reduce_sum(att_weights * memory, axis=1)

        return att


# 测试 注意力是否正常运行
#tf.reset_default_graph()
#memory = tf.random_normal(shape=[32,10,100]) # bs * seq_len * hidden_dim
#state = tf.random_normal(shape=[32,100])  # bs * hidden_dim
#mask = tf.cast(tf.random_normal(shape=[32,10]), tf.int32)  # bs * seq_len
#dot_attention(memory,  state, mask)

# 定义解码器
def decoder(encoder_outputs, encoder_state, embedding_matrix, mask,
            cell_size, max_length, y_ph,
            start_token_id=1, keep_prob=1.0,
            teacher_forcing_rate_ph=None,
            use_attention=False, is_train=True):
    """
    encoder_outputs:  bs * seq_len * encoder_cell_size
    encoder_state bs * encoder_cell_size
    embedding_matrix: vocab_size * vocab_dim
    mask : bs * seq_len
    cell_size :
    max_length: 最长长度
    start_token_id:
    keep_prob:
    teacher_forcing_rate_ph:
    use_attention
    is_train
    """
    with tf.variable_scope('decoder'):
        decoder_cell = tf.nn.rnn_cell.GRUCell(
            num_units=cell_size,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='decoder_cell')

        # initial value of out token on previous step in start token
        output_token = tf.ones(shape=(tf.shape(encoder_outputs)[0],), dtype=tf.int32) * start_token_id

        decoder_state = encoder_state

        pred_tokens = []
        logits = []

        for i in range(max_length):
            # teacher forcing
            if i > 0:
                input_token_emb = tf.cond(
                    tf.logical_and(
                        is_train,
                        tf.random_uniform(shape=(), maxval=1) <= teacher_forcing_rate_ph
                    ),
                    lambda : tf.nn.embedding_lookup(embedding_matrix, y_ph[:,i-1]),
                    lambda : tf.nn.embedding_lookup(embedding_matrix, output_token)
                )
            else:
                input_token_emb = tf.nn.embedding_lookup(embedding_matrix, output_token)

            '''
            使用 attn
            '''
            if use_attention:
                att = dot_attention(encoder_outputs, decoder_state, mask, scope='att')
                input_token_emb = tf.concat([input_token_emb, att], axis=1)
            input_token_emb = tf.nn.dropout(input_token_emb, keep_prob = keep_prob)
            # call recurrent cell
            decoder_outputs, decoder_state = decoder_cell(input_token_emb, decoder_state)
            decoder_outputs = tf.nn.dropout(decoder_outputs, keep_prob=keep_prob)

            embeddings_dim = embedding_matrix.get_shape()[1]

            output_proj = tf.layers.dense(decoder_outputs, embeddings_dim, activation=tf.nn.tanh,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='proj', reuse=tf.AUTO_REUSE)

            # compute logits
            output_logits = tf.matmul(output_proj, embedding_matrix, transpose_b=True)

            logits.append(output_logits)
            output_probs = tf.nn.softmax(output_logits)
            output_token = tf.argmax(output_probs, axis=-1)
            pred_tokens.append(output_token)

        y_pred_tokens = tf.transpose(tf.stack(pred_tokens, axis=0),[1,0])
        y_logits = tf.transpose(tf.stack(logits,axis=0),[1,0,2])

    return y_pred_tokens, y_logits


# 测试解码器
#tf.reset_default_graph()
#vocab_size = 100
#hidden_dim = 100
#inputs = tf.cast(tf.random_uniform(shape=[32,1]) * vocab_size, tf.int32)  # bs * seq_len
#mask = tf.cast(tf.random_uniform(shape=[32,10]) * 2, tf.int32)   # bs * seq_len
#inputs_len = tf.reduce_sum(mask, axis=1)
#embedding_matrix = tf.random_uniform(shape=[vocab_size, hidden_dim])
#
#teacher_forcing_rate = tf.random_uniform(shape=())
#y = tf.cast(tf.random_uniform(shape=[32,10]) * vocab_size, tf.int32)
#
#encoder_outputs, encoder_state = encoder(inputs, inputs_len, embedding_matrix, hidden_dim)

#y_pred_tokens, y_logits = decoder(encoder_outputs, encoder_state, embedding_matrix, mask, hidden_dim, max_length=10,
#        y_ph=y, teacher_forcing_rate_ph=teacher_forcing_rate)

#sess = tf.Session()

#print(y_pred_tokens)
#print(y_logits)
#sess.run(tf.global_variables_initializer())

#y_pred, y_last = sess.run([y_pred_tokens,y_logits],
#         feed_dict={})
#print(y_pred)
#print(y_last)
#
from deeppavlov.core.models.tf_model import TFModel


from tensorflow.python.ops import variables

class TFModel():
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'sess'):
            raise RuntimeError('Your tensorflow model {} must'
                               'has sess attribute!'.format(self.__class__.__name__))
        #super().__init__(*args, **kwargs)

    def load(self, exclude_scopes=['Optimizer']):
        '''
        从指定的path加载model
        '''
        #path = str(self.load_path.resolve())
        path = self.load_path
        if tf.train.checkpoint_exists(path):
            var_list = self._get_saveable_variables(exclude_scopes)
            saver = tf.train.Saver(var_list)
            saver.restore(self.sess, path)

    def save(self, exclude_scopes=['Optimizer']):
        """
        保存model到指定的目录
        """
        path = str(self.save_path.resolve())
        var_lst = self._get_saveable_variables(exclude_scopes)
        saver = tf.train.Saver(var_list)
        saver.save(self.sess, path)

    def _get_saveable_variables(self, exclude_scopes=[]):
        all_vars = variables._all_saveable_objects()
        vars_to_train = [var for var in all_vars if all(sc not in var.name for sc in exclude_scopes)]
        return vars_to_train

    def _get_trainable_variables(self, exclude_scopes=[]):
        all_vars = tf.global_variables()
        vars_to_train = [var for var in all_vars if all(sc not in var.name for sc in exclude_scopes)]
        return vars_to_train

    def get_train_op(self,
                     loss,
                     learning_rate,
                     optimizer=None,
                     clip_norm=None,
                     learnable_scopes=None,
                     optimizer_scope_name=None):
        """
        """
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
                            variable_to_train.append(var)

            if optimizer is None:
                optimizer = tf.train.AdamOptimizer

            # for batch norm it is necesary to update running average
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(extra_update_ops):
                opt = optimizer(learning_rate)
                grads_and_vars = opt.compute_gradients(loss, var_list=variables_to_train)

                if clip_norm is not None:
                    grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var)
                                      for grad, var in grads_and_vars]

                train_op = opt.apply_gradients(grads_and_vars)

        return train_op



@register('seq2seq')
class Seq2Seq(TFModel):

    def __init__(self, **kwargs):
        # hyper parameters

        # dimension of word embeddings
        self.embeddings_dim = kwargs.get('embeddings_dim',100)
        # size of recurrent cell in encoder and decoder
        self.cell_size = kwargs.get('cell_size', 200)
        # dropout keep_probability
        self.keep_prob = kwargs.get('keep_prob', 0.8)
        # learning rate
        self.learning_rate = kwargs.get('learning_rate',3e-05)
        self.max_length = kwargs.get('max_length', 20)
        self.grad_clip = kwargs.get('grad_clip', 5.0)
        self.start_token_id = kwargs.get('start_token_id', 1)
        self.vocab_size = kwargs.get('vocab_size', 11595)
        self.teacher_forcing_rate = kwargs.get('teacher_forcing_rate',0.0)
        self.use_attention = kwargs.get('use_attention',False)

        # create tensorflow session to run computational graph in it
        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)

        self.init_graph()

        self.train_op = self.get_train_op(self.loss, self.lr_ph,
                                          optimizer=tf.train.AdamOptimizer,
                                          clip_norm=self.grad_clip)

        # initialize graph variables
        self.sess.run(tf.global_variables_initializer())

        super().__init__(**kwargs)

        #self.load_path = "./model/seq2seq"

        #if self.load_path is not None:
        #    self.load()

    def init_graph(self):
        #
        self.init_placeholders()

        self.x_mask = tf.cast(self.x_ph, tf.int32)
        self.y_mask = tf.cast(self.y_ph, tf.int32)

        self.x_len = tf.reduce_sum(self.x_mask, axis=1)

        self.embeddings = tf.Variable(tf.random_uniform((self.vocab_size, self.embeddings_dim),-0.1,0.1,
                                                        name='embeddings'), dtype=tf.float32)

        # encoder
        encoder_outputs, encoder_state = encoder(self.x_ph, self.x_len, self.embeddings, self.cell_size, self.keep_prob)

        # decoder
        self.y_pred_tokens, y_logits = decoder(encoder_outputs, encoder_state, self.embeddings, self.x_mask,
                                               self.cell_size, self.max_length,
                                               self.y_ph, self.start_token_id, self.keep_prob_ph,
                                               self.teacher_forcing_rate_ph, self.use_attention,
                                               self.is_train_ph)

        # loss
        self.y_ohe = tf.one_hot(self.y_ph, depth=self.vocab_size)
        self.y_mask = tf.cast(self.y_mask, tf.float32)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_ohe, logits=y_logits) * self.y_mask

        self.loss = tf.reduce_sum(self.loss) / tf.reduce_sum(self.y_mask)


    def init_placeholders(self):
        self.x_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='x_ph')
        self.y_ph = tf.placeholder_with_default(tf.zeros_like(self.x_ph), shape=(None,None),name='y_ph')

        # placeholders for model parameters
        self.lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name='lr_ph')
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph')
        self.is_train_ph = tf.placeholder_with_default(False, shape=[], name='is_train_ph')
        self.teacher_forcing_rate_ph = tf.placeholder_with_default(0.0, shape=[], name='teacher_forcing_rate_ph')

    def _build_feed_dict(self, x,y=None):
        feed_dict = {
            self.x_ph: x,
        }

        if y is not None:
            feed_dict.update({
                self.y_ph: y,
                self.lr_ph: self.learning_rate,
                self.keep_prob_ph: self.keep_prob,
                self.is_train_ph: True,
                self.teacher_forcing_rate_ph: self.teacher_forcing_rate})
        return feed_dict

    def train_on_batch(self,x, y):
        feed_dict = self._build_feed_dict(x, y)
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

    def __call__(self, x):
        feed_dict = self._build_feed_dict(x)
        y_pred = self.sess.run(self.y_pred_tokens, feed_dict=feed_dict)

        return y_pred


s2s = Seq2Seq(
    #save_path="./model/seq2seq",
    #load_path="./model/seq2seq"
)

s2s.sess.run(tf.global_variables_initializer())

iterator = PersonCharChatIterator(data)

epoch_count = 0
for x_batch, y_batch in iterator.gen_batches(32,'train'):
    X_batch = vocab([x.split(" ") for x in x_batch])
    Y_batch = vocab([y.split(" ") for y in y_batch])

    X_batch_max_len = max([len(x) for x in X_batch])
    Y_batch_max_len = max([len(y) for y in Y_batch])
    #X_batch_max_len = max([len(x) for x in X_batch])
    Y_batch_max_len = 20

    x_padder = SentencePadder(length_limit=X_batch_max_len)
    y_padder = SentencePadder(length_limit=18)

    X_batch = np.asarray(x_padder([x for x in X_batch]))
    Y_batch = np.asarray(y_padder([y for y in Y_batch]))

    print(X_batch.shape)
    print(Y_batch.shape)

    # 进行训练
    loss = s2s.train_on_batch(X_batch,Y_batch)

    print("Epoch %d , loss: %f" % (epoch_count, loss))
    epoch_count += 1









#print(vocab(s2s(padder(vocab([['hello', 'my', 'friend', 'there_is_no_such_word_in_dataset', 'and_this']])))) )



#print(iterator.get_instances(data_type='train')[0])
def train():
    """
    测试 batch load
    """
#    print iterator.get_instances(data_type='train')[0]

    #vocab.fit(tokenizer(iterator.get_instances(data_type='train')[0]), tokenizer(iterator.get_instances(data_type='train')[1]))
    pass


def main():
    train()


if __name__ == '__main__':
    main()










