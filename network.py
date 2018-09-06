# -*- coding: utf-8 -*-

"""
策略网络
"""
import copy
from typing import Dict
import collections
import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav

class GoalOrientedBotNetwork(TFModel):
    """
    """
    GRAPH_PARAMS = ["hidden_size", "action_size", "dense_size", "obs_size",
                    "attention_mechanism"]

    def __init__(self,
                 hidden_size,
                 action_size,
                 obs_size,
                 learning_rate,
                 end_learning_rate,
                 decay_steps=1000,
                 decay_power=1.,
                 dropout_rate=0.,
                 l2_reg_coef=0.,
                 dense_size=None,
                 optimizer = 'AdamOptimizer',
                 attention_mechanism=None,   # dict
                 **kwargs):
        end_leanring_rate = end_learning_rate or leanring_rate
        dense_size = dense_size or hidden_size

        # 解析出模型参数
        self.opt = {
            'hidden_size': hidden_size,
            'action_size': action_size,
            'obs_size': obs_size,
            'dense_size': dense_size,
            'learning_rate': learning_rate,
            'end_learning_rate': end_learning_rate,
            'decay_steps': decay_steps,
            'decay_power': decay_power,
            'dropout_rate': dropout_rate,
            'l2_reg_coef': l2_reg_coef,
            'optimizer': optimizer,
            'attention_mechanism': attention_mechanism
        }


        # 初始化 网络参数
        self._init_params()
        # 构建计算图
        self._build_graph()
        # 初始化会话
        self.sess = tf.Session()

        # compile
        self.sess.run(tf.global_variables_initializer())

        super().__init__(**kwargs)

        if tf.train.checkpoint_exists(str(self.save_path.resolve())):
            print('[initializing `{}` for save]'.format(self.__class__.__name__))
            self.load()
        else:
            print("[intializing `{}` from scratch]".format(self.__class__.__name__))

        self.reset_state()

    def __call__(self, features, emb_context, key, action_mask, prob=False):
        feed_dict = {
            self._features: features,
            self._dropout_keep_prob: 1.,
            self._learning_rate: 1.,
            self._utterance_mask: [[1.]],
            self._initial_state: (self.state_c, self.state_h),
            self._action_mask: action_mask
        }

        if self.attn:
            feed_dict[self._emb_context] = emb_context
            feed_dict[self._key] = key

        probs, prediction, state =\
            self.sess.run([self._probs, self._prediction, self._state],
                          feed_dict=feed_dict)

        self.state_c, self._state_h = state

        if prob:
            return probs
        return prediction

    def train_on_batch(self, features, emb_context, key, utter_mask, action_mask, action):
        feed_dict = {
            self._dropout_keep_prob: 1 - self.dropout_rate,
            self._learning_rate: self.get_learning_rate(),
            self._utterance_mask: utter_mask,
            self._features: features,
            self._action: action,
            self._action_mask: action_mask
        }

        if self.attn:
            feed_dict[self._emb_context] = emb_context
            feed_dict[self._key] = key

        _, loss_value,prediction = \
            self.sess.run([self.train_op, self._loss, self._prediction],
                          feed_dict = feed_dict)

        return loss_value, prediction

    def __init_params(self):
        self.learning_rate = self.opt['learning_rate']
        self.end_learning_rate = self.opt['end_learning_rate']
        self.decay_steps = self.opt['decay_steps']
        self.dropout_rate = self.opt['dropout_rate']
        self.hidden_size = self.opt['hidden_size']
        self.action_size = self.opt['action_size']
        self.obs_size = self.opt['obs_size']
        self.dense_size = self.opt['dense_size']
        self.l2_reg = self.opt['l2_reg_coef']

        self._optimizer = None

        if hasattr(self.train, self.opt['optimizer']):
            self._optimizer = getattr(tf.train, self.opt["optimizer"])
        if not issubclass(self._optimizer, tf.train.Optimizer):
            raise ConfigError("`optimizer` parameter should be a name of "
                              " tf.train.Optimizer subclass")

        attn = self.opt.get('attention_mechanism')
        if attn:
            self.opt['attention_mechanism'] = attn
            self.attn = collections.namedtuple('attention_mechanism', attn.keys())(**attn)
            self.obs_size -= attn['token_size']
        else:
            self.attn = None

    def _build_graph(self):
        self._add_placeholders()

        # 初始化body
        _logits, self._state = self._build_body()

        #
        _logits_exp = tf.multiply(tf.exp(_logits), self._action_mask)
        _logits_exp_sum = tf.expand_dims(tf.reduce_sum(_logits_exp,-1),-1)
        self._probs = tf.squeeze(_logits_exp / _logits_exp_sum, name='probs')

        # loss, train and predict operations

        self._prediction = tf.argmax(self._probs, axis=-1, name='prediction')
        _weights = tf.expand_dims(self._utterance_mask,-1)

        _loss_tensor = tf.losses.sparse_softmax_cross_entropy(
            logits=_logits, labels=self._action, weights=_weights,
            reduction=tf.losses.Reduction.NONE)



        self._loss = tf.reduce_mean(_loss_tensor, name='loss')
        self._loss += self.l2_reg * tf.losses.get_regularization_loss()

        self._train_op = self.get_train_op(self._loss,
                                           learning_rate=self._learning_rate,
                                           optimizer=self._optimizer,
                                           clip_norm=2.)

    def _add_placeholders(self):
        self._dropout_keep_prob = tf.placeholder_with_default(1.0,
                                                              shape=[],
                                                              name='dropout_prob')

        self._learning_rate = tf.placeholder(tf.float32,
                                             shape=[],
                                             name='learning_rate')

        self._features = tf.placeholder(tf.float32,
                                        [None,None,self.obs_size],
                                        name='features')

        self._action = tf.placeholder(tf.int32,
                                      [None, None],
                                      name='ground_truth_action')
        self._action_mask = tf.placeholder(tf.float32,
                                           [None, None, self.action_size],
                                           name='action_mask')
        self._utterance_mask = tf.placeholder(tf.float32,
                                              shape=[None, None],
                                              name='utterance_mask')


        _batch_size = tf.shape(self._features)[0]

        zero_state = tf.zeros([_batch_size, self.hidden_size], dtype=tf.float32)

        _initial_state_c = \
            tf.placeholder_with_default(zero_state, shape=[None, self.hidden_size])

        _initial_state_h = \
            tf.placeholder_with_default(zero_state, shape=[None, self.hidden_size])

        self._initial_state = tf.nn.rnn_cell.LSTMStateTuple(_initial_state_c,
                                                            _initial_state_h)

        if self.attn:

            _emb_context_shape = \
                [None,None,self.attn.max_num_tokens, self.attn.token_size]
            self._emb_context = tf.placeholder(tf.float32,
                                               _emb_context_shape,
                                               name='emb_context')

            self._key = tf.placeholder(tf.float32,
                                       [None, None, self.attn.key_size],
                                       name='key')


    def _build_body(self):
        """
        说明 输入分成两个部分一个是 句子级别的编码 self._features
        一个是 attn 级别的编码： self.key 和 self._emb_context

        body构建 使用 线性映射 处理 self._features
        使用注意力机制 编码 self._emb_context
        然后将上面两个特征拼接起来, 输入到 rnn ， 进行action分类
        """
        _units = tf.layers.dense(self._features, self._dense_size,
                                 kernel_regularizer=tf.nn.l2_loss,
                                 kernel_initializer=xav())

        if self.attn:
            attn_scope = 'attention_mechanism/{}'.format(self.attn.type)
            with tf.variable_scope(attn_scope):
                if self.attn.type == 'general':
                    _attn_output = am.general_attention(
                        self.key,
                        self._emb_context,
                        hidden_size=self.attn.hidden_size,
                        projected_align=self.attn.projected_align)

                elif self.attn.typ == 'bahdanau':
                    _attn_output = am.bahdanau_attention(
                        self._key,
                        self._emb_context,
                        hidden_size=self.attn.hidden_size,
                        projected_align = self.attn.projected_align)
                elif self.attn.type == 'cs_general':
                    _attn_output = am.cs_general_attention(
                        self._key,
                        self._emb_context,
                        hidden_size = self.attn.hidden_size,
                        depth = self.attn.depth,
                        projected_align=self.attn.projected_align)
                elif self.attn.type=='cs_bahdanau':
                    _attn_output = am.cs_bahdanau_attention(
                        self._key,
                        self._emb_context,
                        hidden_size = self.attn.hidden_size,
                        depth=self.attn.depth,
                        projected_align=self.attn.projected_align
                    )
                elif self.attn.type=='light_general':
                    _attn_output = am.light_general_attention(
                        self._key,
                        self._emb_context,
                        hidden_size = self.attn.hidden_size,
                        projected_align=self.attn.projected_align)
                elif self.attn.type == 'light_bahdanau':
                    _attn_output = am.light_bahdanau_attention(
                        self._key,
                        self._emb_context,
                        hidden_size=self.attn.hidden_size,
                        projected_align = self.attn.projected_align)
                else:
                    raise ValueError("wrong value for attenton mechanism type")

            _units = tf.concat([_units, _attn_output],-1)

        _units = tf_layers.variational_dropout(_units,
                                               keep_prob=self._dropout_keep_prob)

        # 使用 rnn 进行句子级别的总结
        _lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        _utter_lengths = tf.to_int32(tf.reduce_sum(self._utterance_mask, axis=-1))
        _output, _state = tf.nn.dynamic_rnn(_lstm_cell,
                                            _units,
                                            initial_state = self._initial_state,
                                            sequence_length=_utter_lengths)

        # 注意这里一次性处理一个dialogue的预测
        # 在真实infer 处理中间阶段的 预测时，是保留初始状态实现的
        _logits = tf.layers.dense(_output, self.action_size,
                                  kernel_regularizer=tf.nn.l2_loss,
                                  kernel_initializer=xav(), name='logits')

        return _logits, _state

    def get_learning_rate(self):
        # polynomial decay
        global_step = min(self.global_step, self.decay_steps)

        decay_learning_rate = \
            (self.learning_rate - self.end_learning_rate) * \
            (1 - global_step / self.decay_steps) ** self.decay_power + \
            self.end_learing_rate

        return decayed_learning_rate


    def load(self, *arg, **kwargs):
        self.load_params()
        super().load(*args, **kwargs)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        self.save_params()

    def save_params(self):
        path = str(self.save_path.with_suffix('.json').resolve())
        print('[saving parameter to {}]'.format(path))

        with open(path, 'w', encoding='utf-8') as fp:
            json.dump(self.opt,fp)

    def load_params(self):
        path = str(self.load_path.with_suffix('.json').resolve())
        print('[loading parameters from {}]'.format(path))
        with open(path,'r') as fp:
            parms = json.load(fp)

        for p in self.GRAPH_PARAMS:
            if self.opt.get(p) != params.get(p):
                raise ConfigError("`{}` parameter must be equal to save model"
                                  "parameter value `{}`, but  is euqal to `{}`".format(
                                  p,params.get(p), self.opt.get(p)))

    def process_event(self, event_name, data):
        if event_name == 'after_epoch':
            print("Updating global step, learning rate = {:.6f}".format(self.get_learning_rate()))
            self.global_step += 1

    def reset_state(self):
        # 将网络状态重置为 o
        self.state_c = np.zeros([1, self.hidden_size], dtype=np.float32)
        self.state_h = np.zeros([1, self.hidden_size], dtype=np.float32)

        # 将全局 step 设置为0
        self.global_step = 0

    def shutdown(self):
        self.sess.close()
