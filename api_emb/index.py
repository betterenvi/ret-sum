"""Index."""
import logging

import numpy as np


class Indexer(object):
    """Indexer."""

    def __init__(self, dict_path, min_cnt=1, word2id_dict=None,
                 name='default', eos_wht=''):
        """Init.

        dict_path: a chinese file used to initialize chinese word_id dict
        """
        super().__init__()
        assert (dict_path is None and word2id_dict is None) is False

        self.SOS = '<sos>'
        self.EOS = '<eos>'
        self.UNK = '<unk>'

        self._dict_path = dict_path
        self._min_cnt = min_cnt
        self._name = name
        self._eos_wht = eos_wht
        self._build_dict(word2id_dict)

    def _build_dict(self, word2id_dict):
        """
        Build Chinese dictionary by loading dict file, map each wordacter to idx
        """
        self._id2weight_dict = {}

        if word2id_dict is None:
            self._word2id_dict = {}

            idx = 0
            for k in [self.SOS, self.EOS, self.UNK]:
                self._word2id_dict[k] = idx
                idx += 1

            with open(self._dict_path) as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.strip()
                    fileds = line.split('\t')
                    word = fileds[0]
                    cnt = int(fileds[1])
                    wgt = float(fileds[2])
                    if cnt < self._min_cnt:
                        logging.info('Stop reading more words with cnt < {} '
                                     'when building {} indexer.'
                                     .format(self._min_cnt, self._name))
                        break

                    if word not in self._word2id_dict.keys():
                        self._word2id_dict[word] = idx
                        self._id2weight_dict[idx] = wgt
                        idx += 1
        else:
            self._word2id_dict = word2id_dict

        self._id2word_dict = dict(zip(
            self._word2id_dict.values(),
            self._word2id_dict.keys()))

        median_wgt = np.median(list(self._id2weight_dict.values()))
        self._id2weight_dict[self._word2id_dict[self.SOS]] = 0.
        if self._eos_wht == '':
            self._id2weight_dict[self._word2id_dict[self.EOS]] = 0.
        elif self._eos_wht == 'median':
            self._id2weight_dict[self._word2id_dict[self.EOS]] = median_wgt
        else:
            raise NotImplementedError()
        self._id2weight_dict[self._word2id_dict[self.UNK]] = median_wgt

        self._num_words = len(self._word2id_dict.keys())

    @property
    def num_words(self):
        return self._num_words

    def get_all_words(self):
        return self._id2word_dict.values()

    def id2weight(self, idx):
        return self._id2weight_dict.get(
            idx, self._id2weight_dict.get(self.word2id(self.UNK)))

    def ids2weights(self, ids):
        return [self.id2weight(idx) for idx in ids]

    def word2id(self, word):
        return self._word2id_dict.get(word, self._word2id_dict.get(self.UNK))

    def id2word(self, idx):
        return self._id2word_dict.get(idx, self.UNK)

    def get_word2id_dict(self):
        return self._word2id_dict

    def text2ids(self,
                 text,
                 add_sos=False,
                 add_eos=False,
                 num_steps=None):
        """Return a list."""
        words = text if type(text) is list else text.split()

        text_len = len(words)
        num_steps = num_steps or (add_sos + text_len + add_eos)

        if add_sos:
            ids = [self.word2id(self.SOS)]
        else:
            ids = []

        ids += [self.word2id(word) for word in words]
        ids += [self.word2id(self.EOS)] * (num_steps - len(ids))

        return ids
