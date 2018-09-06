# -*- coding: utf-8 -*-

"""
读取 对话数据集
"""

import copy
import json
from pathlib import Path
from typing import Dict, List

from overrides import overrides

class DSTC2DatasetReader(object):
    """ """
    def _data_fname(self,datatype):
        assert datatype in ('trn', 'val', 'tst'), 'wrong datatype name'
        return 'dstc2-{}.jsonlist'.format(datatype)


    def read(self, data_path, dialogs):
        '''
        读取数据集
        '''
        requred_files = (self._data_fname(dt) for dt in ('trn', 'val', 'tst'))
        #if not all(Path(data_path,f).exists() for f in requred_files):

        data = {
            'train': self._read_from_file(
                Path(data_path, self._data_fname('trn')), dialogs),
            'valid': self._read_from_file(
                Path(data_path, self._data_fname('val')), dialogs),
            'test': self._read_from_file(
                Path(data_path, self._data_fname('tst')), dialogs)
        }

        return data

    @classmethod
    def _read_from_file(cls, file_path, dialogs=False):
        """  """

        utterances, responses, dialogs_indices = \
            cls._get_turns(cls._iter_file(file_path), with_indices=True)

        data = list(map(cls._format_turn, zip(utterances, responses)))

        if dialogs:
            return [data[idx['start']: idx['end']] for idx in dialogs_indices]
        return data

    @staticmethod
    def _format_turn(turn):
        x = {'text': turn[0]['text'],
             'intents': turn[0]['dialog_acts']}
        if turn[0].get('db_result') is not None:
            x['db_result'] = turn[0]['db_result']
        if turn[0].get('episode_done'):
            x['episode_done'] = True

        y = {'text': turn[1]['text'],
             'act': turn[1]['dialog_acts'][0]['act']}

        return (x,y)

    @staticmethod
    def _iter_file(file_path):
        #path = file_path.resolve()
        #fin = open(path)
        #path = './tmp/my_download_of_dstc2/dstc2-trn.jsonlist'
        for ln in open(file_path, 'rt'):
            if ln.strip():
                yield json.loads(ln)
            else:
                yield {}

    @staticmethod
    def _get_turns(data, with_indices=False):
        utterances = []
        responses = []
        dialogs_indices = []

        n = 0
        num_dialog_utter, num_dialog_resp = 0,0

        episode_done = True
        for turn in data:
            if not turn:
                if num_dialog_utter != num_dialog_resp:
                    raise RuntimeError("Datafile in the wrong format")
                episode_done = True
                n += num_dialog_utter
                dialogs_indices.append({
                    'start': n - num_dialog_utter,
                    'end': n,
                })

                num_dialog_utter, num_dialog_resp = 0,0

            else:
                speaker = turn.pop('speaker')
                if speaker == 1:
                    if episode_done:
                        turn['episode_done'] = True
                    utterances.append(turn)
                    num_dialog_utter += 1

                elif speaker == 2:
                    if num_dialog_utter - 1 == num_dialog_resp:
                        responses.append(turn)
                    elif num_dialog_utter -1 < num_dialog_resp:
                        if episode_done:
                            responses.append(turn)
                            utterances.append({
                                "text": "",
                                "dialog_acts": [],
                                "episode_done":True})
                        else:
                            new_turn = copy.deepcopy(utterances[-1])
                            if 'db_result' not in responses[-1]:
                                raise RuntimeError("Every api_call action should have\
                                                   db_result, turn ={}".format(responses[-1]))
                            new_turn['db_result'] = responses[-1].pop('db_result')
                            utterances.append(new_turn)
                            responses.append(turn)

                        num_dialog_utter += 1
                    else:
                        raise RuntimeError("there cannot be two sucessive turns of speaker 1")
                    num_dialog_resp += 1
                else:
                    raise RuntimeError("Only speaker 1 and 2 are supported")
                episode_done = False

        if with_indices:
            return utterances, responses, dialogs_indices
        return utterances, reponses


def main():
    '''
    测试数据集加载
    '''
    dataset_path = "./tmp/my_download_of_dstc2"
    dstc_reader = DSTC2DatasetReader()

    dataset = dstc_reader.read(data_path=dataset_path,dialogs = True)

    print(len(dataset['train']))
    print(len(dataset['valid']))
    print(len(dataset['test']))

    print(dataset['valid'][0][1])
    print(dataset['valid'][1][1])
    print(dataset['valid'][2][1])

    for dialog in dataset['train']:
        print('=='*50)
        for turn in dialog:
            print('--'* 30)
            print(turn)

        break


if __name__ == '__main__':
    main()





