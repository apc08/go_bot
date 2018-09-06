# -*- coding: utf-8 -*-

"""
tracker
"""

import numpy as np

class Tracker():
    """
    tracker 基类
    """
    @abstractmethod
    def reset_state(self):
        """
        重置状态
        """
        pass

    @abstractmethod
    def update_state(self, slots):
        """
        根据当前轮更新状态
        """
        pass

    @abstractmethod
    def get_state(self):
        """
        返回最新的状态
        """
        pass

    @abstractmethod
    def get_feaetures(self):
        """
        返回当前状态的编码
        """
        pass

class DefaultTracker(Tracker):
    """ """
    def __init__(self, slot_names):
        self.slot_names = list(slot_names)
        self.reset_state()

    @property
    def state_size(self):
        return len(self.slot_names)

    @property
    def num_features(self):
        return self.state_size

    def reset_state(self):
        self.history = []
        self.curr_feats = np.zeros(self.num_features, dtype=np.float32)

    def update_state(self,slots):
        def _filter(slots):
            return filter(lambda s: s[0] in self.slot_names, slots)

        if type(slots) == list:
            self.history.extend(_filter(slots))
        elif type(slots) == dict:
            for slot,value in _filter(slots.items()):
                self.history.append((slot,value))
        self.curr_feats = self._binary_features()

        return self

    def get_state(self):
        lasts = {}
        for slot, value in self.history:
            lasts[slot] = value
        return lasts

    def _binary_features(self):
        feats = np.zeros(self.state_size,dtype=np.float32)
        lasts = self.get_state()
        for i, slot in enumerate(self.slot_names):
            if slot in lasts:
                feats[i] = 1.
        return feats

    def get_features(self):
        return self.curr_feats


class FeaturizedTracker(Tracker):
    """ """
    def __init__(self,slot_names):
        self.slot_names = list(slot_names)
        self.reset_state()

    @property
    def state_size(self):
        return len(self.slot_names)

    @property
    def num_features(self):
        return self.state_size * 3 + 3

    def reset_state(self):
        self.history = []
        self.curr_feats = np.zeros(self.num_features, dtype=np.float32)

    def update_state(self, slots):
        def _filter(slots):
            return filter(lambda s: s[0] in self.slot_names, slots)
        prev_state = self.get_state()
        if type(slots) == list:
            self.history.extend(_filter(slots))
        elif type(slots) == dict:
            for slot, value in _filter(slots.items()):
                self.history.append((slot, value))
        bin_feats = self._binary_features()
        diff_feats = self._diff_features(prev_state)
        new_feats = self._new_features((bin_feats,
                                        diff_feats,
                                        np.sum(bin_feats),
                                        np.sum(diff_feats),
                                        np.sum(new_feats)))

        return self

    def get_state(self):
        lasts = {}
        for slot, value in self.history:
            lasts[slot] = value
        return lasts

    def _binary_features(self):
        feats = np.zeros(self.state_size, dtype=np.float32)
        lasts = self.get_state()

        for i, slot in enumerate(self.slot_names):
            if slot in lasts:
                feats[i] = 1.
        return feats

    def _diff_features(self, state):
        feats = np.zeros(self.state_size, dtype=np.float32)
        curr_state = self.get_state()
        for i, slot in enumerate(self.slot_names):
            if (slot in curr_state) and (slot in state) and \
                    (curr_state[slot] != state[slot]):
                feats[i] = 1.
        return feats

    def _new_features(self,state):
        feats = np.zeros(self.state_size, dtype=np.float32)
        curr_state = self.get_state()
        for i, slot in enumerate(self.slot_names):
            if (slot in curr_state) and (slot not in state):
                feats[i] = 1.
        return feats

    def get_features(self):
        return self.curr_feats
