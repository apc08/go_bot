# -*- coding: utf-8 -*-

"""
数据iterator
"""

# 数据遍历器基类

from random import Random
from typing import List, Dict, Tuple, Any, Iterator
from sklearn.model_selection import train_test_split

class DataLearningIterator:

    def split(self, *args, **kwargs):
        pass

    def __init__(self, data, seed, shuffle, *args, **kwargs):
        self.shuffle = shuffle
        self.random = Random(seed)

        self.train = data.get('train', [])
        self.valid = data.get('valid', [])
        self.test  = data.get('test', [])
        self.split(*args, **kwargs)

        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self.train + self.test + self.valid
        }


    def get_batches(self, batch_size, data_type='train', shuffle=None):
        """
        """
        if shuffle is None:
            shuffle = self.shuffle

        data = self.data[data_type]
        data_len = len(data)

        if data_len == 0:
            return

        order = list(range(data_len))

        if shuffle:
            self.random.shuffle(order)

        if batch_size < 0:
            batch_size = data_len

        for i in range((data_len - 1) // batch_size - 1):
            yield tuple(zip(*[data[o] for o in order[i * batch_size: (i+1) * batch_size]]))

    def get_instances(self, data_type):
        '''
        '''
        data = self.data[data_type]
        return tuple(zip(*data))


# dialog iterator
class DialogDatasetIterator(DataLearningIterator):
    """
    """

    @staticmethod
    def _dialogs(data):
        dialogs = []
        prev_resp_act = None
        for x, y in data:
            if x.get('episode_done'):
                del x['episode_done']
                prev_resp_act = None
                dialogs.append(([], []))
            x['prev_resp_act'] = prev_resp_act
            prev_resp_act = y['act']
            dialogs[-1][0].append(x)
            dialogs[-1][1].append(y)

        return dialogs

    #@overrides
    def split(self, *args, **kwargs):
        self.train = self._dialogs(self.train)
        self.valid = self._dialogs(self.valid)
        self.test  = self._dialogs(self.test)

class DialogDBResultDatasetIterator(DataLearningIterator):
    """
    """
    @staticmethod
    def _db_result(data):
        x, y = data
        if 'db_result' in x:
            return x['db_result']

    #@overrides
    def split(self, *args, **kwargs):
        self.train = [(r, "") for r in filter(None, map(self._db_result, self.train))]
        self.valid = [(r, "") for r in filter(None, map(self._db_result, self.valid))]
        self.test  = [(r, "") for r in filter(None, map(self._db_result, self.test))]



# 基本的分类 数据iterator
class BasicClassificationDatasetIterator(DataLearningIterator):
    """
    """
    def __init__(self, data,
                 fields_to_merge,
                 merged_field,
                 field_to_split,
                 split_fields,
                 split_proportions,
                 seed = None,
                 shuffle=True,
                 *args,
                 **kwargs):
        """
        基本分类数据遍历类
        """
        super().__init__(data, seed=seed, shuffle=shuffle)

        if fields_to_merge is not None:
            if merged_field is not None:
                self._merge_data(fields_to_merge=fields_to_merge,
                                 merged_field=merged_field)
            else:
                raise IOError("Given fields to merge BUT not given of merged field")

        if field_to_split is not None:
            if split_fields is not None:

                self._split_data(field_to_split=field_to_split,
                                 split_fields=split_fields,
                                 split_proportions=[float(s) for s in split_proportions])

            else:
                raise IOError("Given field to split BUT not given names of split fields")

    def _split_data(self, field_to_split, split_fields,split_proportions):
        """
        """
        data_to_div = self.data[field_to_split].copy()
        data_size = len(self.data[field_to_split])

        for i in range(len(split_fields)-1):
            self.data[split_fields[i]], data_to_div = train_test_split(
                data_to_div,
                test_size = len(data_to_div) - int(data_size * split_proportions[i]))

            self.data[split_fields[-1]] = data_to_div
        return True

    def _merge_data(self, fields_to_merge, merged_field):
        """
        """
        data = self.data.copy()
        data[merged_field] = []
        for name in fields_to_merge:
            data[merged_field] += self.data[name]
        self.data = data
        return True



# 意图分类
class Dstc2IntentsDatasetIterator(BasicClassificationDatasetIterator):

    def __init__(self,data,
                 fields_to_merge=['train','valid'],
                 merged_field='train',
                 field_to_split='train',
                 split_fields=['train','valid'],
                 split_proportions=[0.9,0.1],
                 seed=42,
                 shuffle=True):
        super().__init__(data, fields_to_merge, merged_field,
                         field_to_split, split_fields, split_proportions,
                         seed=seed, shuffle=shuffle)

        new_data = dict()
        new_data['train'] = []
        new_data['valid'] = []
        new_data['test']  = []

        for field in ['train', 'valid', 'test']:
            for turn in self.data[field]:
                reply = turn[0]
                curr_intents = []
                #print(turn)
                #print(reply)
                if reply['intents']:
                    for intent in reply['intents']:
                        for slot in intent['slots']:
                            if slot[0] == 'slot':
                                curr_intents.append(intent['act'] + '_' + slot[1])
                            else:
                                curr_intents.append(intent['act'] + '_' + slot[0])
                    if len(intent['slots']) == 0:
                        curr_intents.append(intent['act'])
                else:
                    if reply['text']:
                        curr_intents.append('unknown')
                    else:
                        continue
                new_data[field].append((reply['text'], curr_intents))

        self.data = new_data


# ner data iterator

class Dstc2NerDatasetIterator(DataLearningIterator):
    """ """

    def __init__(self, data, dataset_path="", seed=42, shuffle=False):
        self.shuffle = shuffle
        self.random = Random(seed)

        dataset_path = "./tmp/my_download_of_dstc2/dstc_slot_vals.json"
        with open(dataset_path, encoding='utf-8') as f:
            self._slot_vals = json.load(f)
        for data_type in ['train', 'test', 'valid']:
            bio_markup_data = self._preprocess(data.get(data_type,[]))
            setattr(self, data_type, bio_markup_data)

        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test':  self.test,
            'all' :  self.train + self.test + self.valid}

        self.shuffle = shuffle

    def _preprocess(self,data_part):
        processed_data_part = list()
        processed_texts = dict()
        for sample in data_part:
            for utterance in sample:
                if 'intents' not in utterance or len(utterance['text']) < 1:
                    continue
                text = utterance['text']
                intents = utterance.get('intents', dict())
                slots = list()
                for intent in intents:
                    current_slots = intent.get('slots', [])
                    for slot_type, slot_val in current_slots:
                        if slot_type in self._slot_vals:
                            slots.append((slot_type, slot_val,))

            # remove dupliicate pairs (text, slots)

                if (text in processed_texts) and (slots in processed_texts[text]):
                    continue
                processed_texts[text] = processed_texts.get(text, []) + [slots]

                processed_data_part.append(self._add_bio_markup(text, slots))

        return processed_data_part

    def _add_bio_markup(self, utterance, slots):
        tokens = utterance.split()
        n_toks = len(tokens)
        tags = ['O' for _ in range(n_toks)]
        for n in range(n_toks):
            for slot_type, slot_val in slots:
                for entity in self._slot_vals[slot_type][slot_val]:
                    slot_tokens = entity.split()
                    slot_len = len(slot_tokens)
                    if n + slot_len <= n_toks and self._is_equal_sequence(tokens[n: n + slot_len],
                                                                           slot_tokens):
                        tags[n] = 'B-' + slot_type
                        for k in range(1, slot_len):
                            tags[n+k] = 'I-' + slot_type
                        break
        return tokens, tags


    @staticmethod
    def _is_equal_sequence(seq1, seq2):
        equality_list = [tok1 == tok2 for tok1, tok2 in zip(seq1, seq2)]

        return all(equality_list)

    @staticmethod
    def _build_slot_vals(slot_vals_json_path='data/'):
        pass




from dstc_reader import *


def main():
    '''
    '''
    # 加载数据集
    dataset_path = "./tmp/my_download_of_dstc2"
    dstc_reader = DSTC2DatasetReader()

    data = dstc_reader.read(data_path=dataset_path, dialogs=False)

    # 测试 intent 加载类

    #intentIterator = Dstc2IntentsDatasetIterator(data)

    #for x_batch,y_batch in intentIterator.get_batches(batch_size=5):
    #    for x,y in zip(x_batch, y_batch):
    #        print("--" * 50)
    #        print(x)
    #        print(y)
        #for ele in batch:
        #    print('--'*40)
        #    print(ele)

    # 测试 ner 数据加载
    nerIterator = Dstc2NerDatasetIterator(data)

    for x_batch, y_batch in nerIterator.get_batches(batch_size=5):

        for x, y in zip(x_batch, y_batch):
            print("--" * 50)
            print(x)
            print(y)

    #print(nerIterator.train)



    # 测试 dialogue 数据加载
    #data = dstc_reader.read(data_path=dataset_path, dialogs=False)
    #dialog_iter = DialogDatasetIterator(data,shuffle=False, seed=42)

    #for x_batch, y_batch in dialog_iter.get_batches(batch_size=32):
    #    for x,y in zip(x_batch, y_batch):
    #        print("==" * 50)
    #        for u,s in zip(x,y):
    #            print("--" * 50)
    #            print(u)
    #            print(s)
#
#        break






if __name__ == '__main__':
    main()

