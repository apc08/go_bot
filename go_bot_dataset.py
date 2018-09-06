# -*- coding: utf-8 -*-

'''
展示 bot 数据集
'''

import json
import copy
from pprint import pprint
import os
import deeppavlov

from deeppavlov.dataset_readers.dstc2_reader import DSTC2DatasetReader

# 下载数据集
data = DSTC2DatasetReader().read(data_path="tmp/my_download_of_dstc2")


# 读取数据集并进行展示
from deeppavlov.dataset_iterators.dialog_iterator import DialogDatasetIterator

batches_generator = DialogDatasetIterator(data, seed=1443, shuffle=True).gen_batches(batch_size=4, data_type='train')


batch = batches_generator.__next__()

x_batch, y_batch = batch

len(x_batch)

dialog_id = 0
dialog = [(x,y) for x,y in zip(x_batch[dialog_id], y_batch[dialog_id])]

turn_id = 0
print("------{}th turn----------".format(turn_id))
pprint(dialog[turn_id],indent=8)

for turn in dialog:
    x,y = turn
    print('::',x['text'])
    print('>>',y['text'],'\n')

'''
配置文件说明

dataset_reader: 数据读取模块配置信息
dataset_iterator: 数据遍历模块配置信息
metadata:  模型的配置信息
train: 训练配置信息
chainer： 数据管道配置信息
'''

vocab_config = {}
dstc2_reader_comp_config = {
    'name': 'dstc2_reader',
    'data_path': 'dstc2'
}

vocab_config['dataset_reader'] = dstc2_reader_comp_config

dialog_iterator_comp_config = {
    'name': 'dialog_iterator'
}

vocab_config['dataset_iterator'] = dialog_iterator_comp_config

# meta data 下载数据url
dstc2_download_config = {
    'url': 'http://files.deeppavlov.ai/datasets/dstc2_v2.tar.gzv',
    'subdir': 'dstc2'
}

vocab_config['metadata'] = {}
vocab_config['metadata']['download'] = [
    dstc2_download_config
]

vocab_config['train'] = {}

vocab_config['chainer'] = {}
vocab_config['chainer']['in'] = ['utterance']
vocab_config['chainer']['in_y'] = []
vocab_config['chainer']['out'] = []


vocab_comp_config = {
    'name': 'default_vocab',
    'save_path': 'vocabs/token.dict',
    'load_path': 'vocabs/token.dict',
    'fit_on':['utterance'],
    'leval':'token',
    'tokenizer': {'name': 'split_tokenizer'},
    'main': True
}

vocab_config['chainer']['pipe'] = [
    vocab_comp_config
]

json.dump(vocab_config, open('./tmp/vocab_config.json','wt'))


from deeppavlov.download import deep_download

deep_download(['--config','./tmp/vocab_config.json'])

dstc2_path = deeppavlov.__path__[0] + '/../download/dstc2'

print(dstc2_path)


from deeppavlov.core.commands.train import train_evaluate_model_from_config
train_evaluate_model_from_config("./tmp/vocab_config.json")

vocabs_path = deeppavlov.__path__[0] + '/../download/vocabs'

vocab_comp_config['in'] = ['utterance']
vocab_comp_config['out'] = ['utterance_token_indices']

vocab_config['chainer']['pipe'] = [
    vocab_comp_config
]

vocab_config['chainer']['out'] = ['utterance_token_indices']

from deeppavlov.core.commands.infer import build_model_from_config

model = build_model_from_config(vocab_config)

print(model(['hi']))


# model 配置
from deeppavlov.download import deep_download
from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.commands.infer import build_model_from_config

simple_config = {}

simple_config['dataset_reader'] = dstc2_reader_comp_config
simple_config['dataset_iterator'] = dialog_iterator_comp_config
simple_config['metadata'] = {}
simple_config['metadata']['download'] = [
    dstc2_download_config
]

simple_config['chainer'] = {}
simple_config['chainer']['in'] = ['x']
simple_config['chainer']['in_y'] = ['y']
simple_config['chainer']['out'] = ['y_predicted']


vocab_comp_config = {
    'name':'default_vocab',
    'id': 'token_vocab',
    'load_path': 'vocabs/token.dict',
    'save_path': 'vocabs/token.dict',
    'fit_on': ['x'],
    'leval': 'token',
    'tokenizer': {'name': 'split_tokenizer'}
}

simple_config['chainer']['pipe'] = []
simple_config['chainer']['pipe'].append(vocab_comp_config)


bot_comp_config = {
    'name': 'go_bot',
    'in': ['x'],
    'in_y': ['y'],
    'out': ['y_predicted'],
    'word_vocab': None,
    'bow_embedder': {"name": "bow"},
    'embedder': None,
    'slot_filler': None,
    'template_path': 'dstc2/dstc2-templates.txt',
    'template_type': 'DualTemplate',
    'database': None,
    'api_call_action': 'api_call',
    'network_parameters': {
      'load_path': 'gobot_dstc2_simple/model',
      'save_path': 'gobot_dstc2_simple/model',
      'dense_size': 64,
      'hidden_size': 128,
      'learning_rate': 0.002,
      'attention_mechanism': None
    },
    'tokenizer': {'name': 'stream_spacy_tokenizer',
                  'lowercase': False},
    'tracker': {'name': 'featurized_tracker',
                'slot_names': ['pricerange', 'this', 'area', 'food', 'name']},
    'main': True,
    'debug': False
}


bot_comp_config['word_vocab'] = '#token_vocab'

slot_filler_comp_config = {
    'config_path': deeppavlov.__path__[0] + '/../deeppavlov/configs/ner/slotfill_dstc2.json'
}

bot_comp_config['slot_filler'] = slot_filler_comp_config

simple_config['chainer']['pipe'].append(bot_comp_config)

simple_bot_train_config = {
    'batch_size': 4,
    'epochs': 2,
    'log_every_n_batches': -1,
    'log_every_n_epochs': 1,
    'metrics': ['per_item_dialog_accuracy'],
    'val_every_n_epochs': 1,
    'validation_patience': 20
}

simple_config['train'] = simple_bot_train_config

json.dump(simple_config, open("gobot/simple_config.json", 'wt'))

deep_download(['--config', slot_filler_comp_config['config_path']])

train_evaluate_model_from_config("gobot/simple_config.json")
