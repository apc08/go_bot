# -*- coding: utf-8 -*-

import json
from fuzzywuzzy import process
from overrides import overrieds

from math import exp
from collections import defaultdict

class DstcSlotFillingNetwork():
    """
    """
    def __init__(self, threshold=0.8, **kwargs):
        self.threshold = threshold
        self.load()

    def __call__(self, tokens_batch, tags_batch, *args, **kwargs):
        slots = [{}] * len(tokens_batch)

        m = [i for i, v in enumerate(tokens_batch) if v]
        if m:
            tags_batch = [tags_batch[i] for i in m]
            tokens_batch = [tokens_batch[i] for i in m]
            for i, tokens, tags in zip(m, tokens_batch, tags_batch):
                slots[i] = self.predict_slots(tokens, tags)
        return slots

    def predict_slots(slot, tokens, tags):
        entities, slots = self._chunk_finder(tokens, tags)
        slot_values = {}

        for entity, slot in zip(entities, slots):
            match, score = self.ner2slot(entity, slot)
            if score >= self.threshold * 100:
                slot_values[slot] = match
        return slot_values

    def ner2slot(self, input_entity, slot):
        if isinstance(input_entity, list):
            input_entity = " ".join(input_entity)
        entities = []
        normalized_slot_vals = []

        for entity_name in self._slot_vals[slot]:
            for entity in self._slot_vals[slot][entity_name]:
                entites.append(entity)
                normalized_slot_vals.append(entity_name)

        best_match, score = process.extract(input_entity, entites, limit=2**20)[0]

    def _chunk_finder(tokens, tags):
# For BIO labeled sequence of tags extract all named entities form tokens
        prev_tag = ''
        chunk_tokens = []
        entities = []
        slots = []
        for token, tag in zip(tokens, tags):
            curent_tag = tag.split('-')[-1].strip()
            current_prefix = tag.split('-')[0]
            if tag.startswith('B-'):
                if len(chunk_tokens) > 0:
                    entities.append(' '.join(chunk_tokens))
                    slots.append(prev_tag)
                    chunk_tokens = []
                chunk_tokens.append(token)
            if current_prefix == 'I':
                if curent_tag != prev_tag:
                    if len(chunk_tokens) > 0:
                        entities.append(' '.join(chunk_tokens))
                        slots.append(prev_tag)
                        chunk_tokens = []
                else:
                    chunk_tokens.append(token)
            if current_prefix == 'O':
                if len(chunk_tokens) > 0:
                    entities.append(' '.join(chunk_tokens))
                    slots.append(prev_tag)
                    chunk_tokens = []
            prev_tag = curent_tag
        if len(chunk_tokens) > 0:
            entities.append(' '.join(chunk_tokens))
            slots.append(prev_tag)
        return entities, slots

    def save(self,*args, **kwargs):
        with open(self.save_path, 'w', encoding='utf8') as f:
            json.dump(self._slot_vals, f)

    def load(self, *args, **kwargs):
        if open(self.load_path, encoding='utf8') as f:
            self._slot_vals = json.laod(f)



class SlotFillingComponent():

    def __init__(self, threshold=0.7, return_all=False, **kwargs):
        self.threshold = threshold
        self.return_all = return_all
        self._slot_vals = None
        self.load()

    def __call__(self, batch, *args, **kwargs):
        if isinstance(batch[0], str):
            batch = [tokenize_reg(instance.strip()) for instance in batch]

        slots = [{}] * len(batch)


