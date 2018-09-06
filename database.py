# -*- coding: utf-8 -*-

from typing import List, Dict
import sqlite3

'''
与数据库交互 接口
'''

class Sqlite3Database():
    '''
    '''
    def __init__(self,
                 save_path,
                 table_name,
                 primary_keys=None,   # list(str)
                 keys=None,      # list(str)
                 unknown_value='UNK',  # str = 'UNK',
                 *args,
                 **kwargs):
        #super().__init__(save_path=save_path, *args, **kweargs)

        self.primary_keys = primary_keys
        #if not self.primary_keys:
        #    raise ValueError('primary keys list can`t be empty')
        self.save_path = save_path
        self.tname = table_name
        self.keys = keys
        self.unknown_value = unknown_value


        self.conn = sqlite3.connect(str(self.save_path), check_same_thread=False)
        self.cursor = self.conn.cursor()

        if self._check_if_table_exists():

            if not self.keys:
                self.keys = self._get_keys()

        else:
            pass

    def __call__(self,
                 batch, # list[Dict]
                 ascending='ASC'): # False
        """
        return shape: list[list[dict]]
        """
        order = "ASC" if ascending else 'DESC'
        print batch
        order_by = 'name'

        if not self._check_if_table_exists():
            # warn:  database is empty call fit before using
            return [[] for i in range(len(batch))]
        return [self._search(b, order_by=order_by, order=order) for b in batch]

    def _check_if_table_exists(self):
        self.cursor.execute("SELECT name FROM sqlite_master "
                            " WHERE type='table' "
                            " AND name='{}';".format(self.tname))
        return bool(self.cursor.fetchall())

    def _search(self, kv, order_by='name', order='ASC'):
        if not kv:
            if order_by is not None:
                self.cursor.execute(" SELECT * FROM {}".format(self.tname) +
                                    " ORDER BY {} {}".format(order_by, order))
            else:
                self.cursor.execute("SELECT * FROM {}".format(self.tname))
        else:
            keys = list(kv.keys())
            values = [kv[k] for k in keys]
            where_expr = ' AND '.join('{}=?'.format(k) for k in keys)

            if order_by is not None:
                self.cursor.execute(" SELECT * FROM {}".format(self.tname) +
                                    " WHERE {}".format(where_expr) +
                                    " ORDER BY {} {}".format(order_by, order),
                                    values)
            else:
                self.cursor.execute("SELECT * FROM {}".format(self.tname) +
                                    " WHERE {}".format(where_expr),
                                    values)

        return [self._wrap_selection(s) for s in self.cursor.fetchall()]

    def _wrap_selection(self, selection):
        if not self.keys:
            self.keys = self._get_keys()

        return {f: v for f, v in zip(self.keys, selection)}

    def _get_keys(self):
        self.cursor.execute("PRAGMA table_info({});".format(self.tname))
        return [info[1] for info in self.cursor]

    def _get_types(self):
        self.cursor.execute("PRAGMA table_info({});".format(self.tname))
        return {info[1]: info[2] for info in self.cursor}

    def fit(self, data):   # data list[Dict]
        if not self._check_if_table_exists():
            self.keys = self.keys or list(set(k for d in data for k in d.keys()))
            self._create_table(self.keys, types)
        elif not self.keys:
            self.keys = self._get_keys()

        self._insert_many(data)

    def _create_table(self, keys, types):
        if any(pk not in keys for pk in self.primary_keys):
            raise ValueError("Primary keys must be from {}".format(keys))
        new_types = { "{} {} primary key".format(k,t) if k in self.primary_keys else
                     " {} {}".format(k,t)
                     for k,t in zip(keys, types)}
        self.cursor.execute("CREATE TABLE IF NOT EXISTS {} ({})".format(
                self.tname, ', '.join(new_types)))

        # create table with keys {} get_types

    def _insert_many(self,data):
        to_insert = {}
        to_update = {}

        for kv in filter(None, data):
            primary_values = tuple(kv[pk] for pk in self.primary_keys)
            record = tuple(kv.get(k, self.unknown_value) for k in self.keys)
            curr_record = self._get_record(primary_values)

            if curr_record:
                if primary_values in to_update:
                    curr_record = to_update[primary_values]
                if curr_record != record:
                    to_update[primary_values] = record
            else:
                to_insert[primary_values] = record
        if to_insert:
            fformat = ",".join(['?'] * len(self.keys))
            self.cursor.executemany("INSERT into {}".format(self.tname) +
                                    " VALUES ({})".format(fformat),
                                    to_insert.values())

        if to_update:
            for record in to_update.values():
                self._update_one(record)

        self.conn.comnit()

    def _get_record(self, primary_values):
        ffields = ', '.join(self.keys) or '*'
        where_expr = ' AND '.join("{}='{}'".format(pk, v)
                                  for pk, v in zip(self.primary_keys, primary_values))

        fetched = self.cursor.execute("SELECT {} FROM {}".format(ffieds, self.tname) +
                                      " WHERE {}".format(where_expr)).fetchone()

        if not fetched:
            return None
        return fetched

    def _update_one(self, record):
        set_expr = ', '.join("{} = '{}'".format(k,v)
                             for k,v in zip(self.keys, record)
                             if k not in self.primary_keys)
        where_expr = ' AND '.join("{} = '{}'".format(k,v)
                                  for k,v in zip(self.keys, record)
                                  if k in self.primary_keys)

        self.cursor.execute("UPDATE {}".format(self.tname) +
                            " SET {}".format(set_expr) +
                            " WHERE {}".format(where_expr))


    def save(self):
        pass

    def load(self):
        pass



# 数据库的 iterator

class SQLiteDataIterator():
    """
    """
    def __init__(self, data_dir='',data_url='',
                 shuffle=None, seed=None, **kwargs):
        #download_dir = expand_path(data_dir)
        #download_path = download_dir.

        self.connect = sqlite3.connect(str(download_path), check_same_thread=False)
        self.db_name = self.get_db_name()
        self.doc_ids = self.get_doc_ids()
        self.doc2index = self.map_dic2idx()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random = Random(seed)

    def get_doc_ids(self):
        '''
        return document ids
        '''
        cursor = self.connect.cursor()
        cursor.execute('SELECT id FROM {}'.format(self.db_name))
        ids = [ids[0] for ids in cursor.fetchall()]
        cursor.close()
        return ids

    def get_db_name(self):
        cursor = self.connect.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        assert cursor.arraysize == 1
        name = cursor.fetchall()[0]
        cursor.close()
        return name

    def map_doc2idx(self):
        '''
        map db ids to integer ids
        '''
        doc2idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        return doc2idx

    def get_doc_content(self, doc_id):
        '''
        '''
        cursor = self.connect.cursor()

        cursor.execute(
            "SELECT text FROM {} WHERE id = ?".format(self.db_name),
            (doc_id,)
        )

        result = cursor.fetchone()
        cursor.close()

        return result if result is None else result[0]


def main():
    '''
    测试 database 的效果

    4 kings parade city centre|moderate|british|01223 365068|c.b 2, 1 s.j|centre|the copper kettle
UNK|expensive|british|01223 359506|UNK|UNK|the cambridge chop house
35 newnham road newnham|expensive|thai|01223 323178|c.b 3, 9 e.y|west|sala thong
15 magdalene street city centre|cheap|italian|01223 315232|c.b 3, 0 a.f|west|la margherita
451 newmarket road fen ditton|moderate|indian|01223 566388|c.b 5, 8 j.j|east|curry prince
sqlite> select * from mytable limit 1;
cambridge leisure park clifton way cherry hinton|cheap|chinese|01223 244277|c.b 1, 7 d.y|south|the lucky star
sqlite> select * from mytable limit 2;
cambridge leisure park clifton way cherry hinton|cheap|chinese|01223 244277|c.b 1, 7 d.y|south|the lucky star
71 castle street city centre|expensive|indian|01223 366668|c.b 3, 0 a.h|west|cocum
sqlite> .tables
mytable
sqlite> pragma table_info('mytable')
   ...> ;
0|addr|text|0||0
1|pricerange|text|0||0
2|food|text|0||0
3|phone|text|0||0
4|postcode|text|0||0
5|area|text|0||0
6|name|text|0||1

    '''
    db_path = "./tmp/my_download_of_dstc2/resto.sqlite"
    tname = 'mytable'
    database = Sqlite3Database(save_path=db_path, table_name=tname)
    kv_dict = {'pricerange':'cheap',
               'food':'chinese',
               'area': 'south'}
    print(database([kv_dict,]))


if __name__ == '__main__':
    main()

