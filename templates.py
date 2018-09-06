# -*- coding: utf-8 -*-

import copy
import re

class Template():

    #@abstractmethod
    def from_str(cls, s):
        return cls(s)


class DefaultTemplate(Template):

    def __init__(self, text=""):
        self.text = text

    @classmethod
    def from_str(cls, s):
        return cls(s)

    def update(self, text=""):
        self.text = self.text or text

    def __contains__(self, t):
        return t.text == self.text

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.text == other.text
        return False

    def __hash__(self):
        return hash(self.text)

    def __str__(self):
        return self.text

    def generate_text(self, slots=[]):
        t = copy.copy(self.text)
        if isinstance(slots, dict):
            slots = slots.items()

        for slot, value in slots:
            t = t.replace('#' + slot, value, 1)
        if t:
            t = t[0].upper() + t[1:]

        return t

class DualTemplate(Template):
    def __init__(self, default="", dontcare=""):
        self.default = default       # default 的模板
        self.dontcare = dontcare     # 不重要slot的模板

    @property
    def dontcare_slots(self):
        default_slots = self._slots(self.default)
        dontcare_slots = self._slots(self.dontcare)
        return default_slots - dontcare_slots

    @staticmethod
    def _slots(text):
        return set(re.findall('#(\w+)',text))

    @classmethod
    def from_str(cls, s):
        return cls(*s.split('\t',1))

    def update(self, default="", dontcare=""):
        self.default = self.default or default
        self.dontcare = dontcare

    def __contains__(self, t):
        return t.default and (t.default == self.default) \
            or t.dontcare and  (tf.dontcare == self.dontcare)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.default == other.default) \
                and (self.dontcare == other.dontcare)

        return False

    def __hash__(self):
        return hash(self.default + '\t' + self.dontcare)

    def __str__(self):
        return self.default + '\t' + self.dontcare

    def generate_text(self, slots):
        t = copy.copy(self.default)
        if isinstance(slots, dict):
            slots = slots.items()

        dontcare_slots = (s[0] for s in slots if s[1] == 'dontcare')
        if self.dontcare and self.dontcare_slots.issubset(dontcare_slots):
            t = copy.copy(self.dontcare)

        for slot, value in slots:
            t = t.replace('#' + slot, value, 1)

        if t:
            t = t[0].upper() + t[1:]

        return t

class Templates:

    def __init__(self, ttype):
        self.ttype = ttype
        self.act2templ = {}
        self.templ2act = {}
        self._actions = []
        self._templates = []


    def __contains__(self, key):
        """
        判断当前的key 是否在template 列表中，
        """
        if isinstance(key, str):
            return key in self.act2templ
        elif isinstance(key, Template):
            return key in self.templ2act

    def __getitem__(self, key):
        '''
        双向索引， 如果 key 是str 那么返回template
        如果 key 是template 返回 act
        '''
        if isinstance(key, str):
            return self.act2templ[key]
        elif isinstance(key, Template):
            return self.templ2act[key]

    def __len__(self):
        return len(self.act2templ)

    def __str__(self):
        return str(self.act2templ)

    def __setitem__(self, key, value):
        key = str(key)
        if key not in self.act2templ:
            self.act2templ[key] = value
            self.templ2act[value] = key
            self._actions = []
            self._templates = []


    @property
    def actions(self):
        if  not self._actions:
            self._actions = sorted(self.act2templ.keys())
        return self._actions

    @property
    def templates(self):
        if not self._templates:
            self._templates = [self.act2templ[a] for a in self.actions]
        return self._templates

    def load(self, filename):
        with open(filename, 'r', encoding='utf8') as fp:
            for ln in fp:
                act, template = ln.strip('\n').split('\t',1)
                self.__setitem__(act, self.ttype.from_str(template))
        return self

    def save(self, filename):
        with open(filename, 'w', encoding='utf8') as outfile:
            for act in sorted(self.actions):
                template = self.__getitem__(act)
                outfile.write('{}\t{}\n'.format(act, template))





def main():
    '''
    测试模板加载
    '''
    template_path = "./tmp/my_download_of_dstc2/dstc2-templates.txt"

    templates = Templates(ttype=DualTemplate)
    templates.load(template_path)
    print(len(templates))
    print(templates.actions)
    print(templates.templates)

    for act, temp in zip(templates.actions,templates.templates):
        print("{} --> {}".format(act, temp))



if __name__ == '__main__':
    main()


