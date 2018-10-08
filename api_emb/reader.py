"""Reader."""
import codecs
import collections
import logging
import os
import pickle

import numpy as np

from copy import deepcopy
from math import ceil

from common.util import unify_len_for_list_of_list

from .tool.codenn_index import code2tokens
from .tool.codenn_index import text2words


class Reader(object):
    """Reader."""

    def __init__(self, config, data_list, api_indexer, doc_indexer):
        super(Reader, self).__init__()
        self.config = config
        self._data_list = data_list
        self._api_indexer = api_indexer
        self._doc_indexer = doc_indexer

    def _parse_line(self, line, api_num_steps, doc_num_steps):
        config = self.config

        cid = None
        _, cid, doc, api, _ = line.split(sep='\t')

        words = text2words(doc, lower=config.lower_doc_idx)     # A list.

        tokens = code2tokens(api, lower=config.lower_api_idx,
                             lang=config.lang)                  # A list.

        add_sos, add_eos = False, True
        api_idxs = self._api_indexer.text2ids(
            tokens, add_sos=add_sos, add_eos=add_eos,
            num_steps=api_num_steps)
        api_seq_len = len(tokens) + add_sos + add_eos

        add_sos, add_eos = False, True
        doc_idxs = self._doc_indexer.text2ids(
            words, add_sos=add_sos, add_eos=add_eos,
            num_steps=doc_num_steps)
        doc_seq_len = len(words) + add_sos + add_eos

        if config.api_config.use_idf:
            api_weights = self._api_indexer.ids2weights(api_idxs)
        else:
            api_weights = [1.] * api_seq_len

        if api_num_steps is not None:
            api_weights += [0.] * (api_num_steps - len(api_weights))

        if config.doc_config.use_idf:
            doc_weights = self._doc_indexer.ids2weights(doc_idxs)
        else:
            doc_weights = [1.] * doc_seq_len

        if doc_num_steps is not None:
            doc_weights += [0.] * (doc_num_steps - len(doc_weights))

        return (api_idxs, api_weights, api_seq_len, tokens,
                doc_idxs, doc_weights, doc_seq_len, words,
                cid)

    def _file_to_pkl_file(self, f, lower_doc_idx):
        if lower_doc_idx:
            return f + '.pkl'
        else:
            return f + '.udoc.pkl'

    def _read_one_epoch(self, data_list=None, batch_all=False,
                        do_pickle=True):
        """Read one epoch of data.

        If batch_all, make all examples as one batch.
        """
        config = self.config
        data_list = self._data_list if data_list is None else data_list

        if do_pickle and config.pickle_data:
            for f in data_list:
                pf = self._file_to_pkl_file(f, config.lower_doc_idx)
                if ((not os.path.exists(pf)) or config.re_pickle or
                        os.path.getmtime(pf) < os.path.getmtime(f)):
                    data = self._to_pickle(f, pf)
                # Load data anyway.
                data = self._from_pickle(f, pf)
                logging.info('Read {} batches in {}'.format(len(data), pf))
                yield from data
        else:
            batch_size = config.batch_size
            api_config = config.api_config
            doc_config = config.doc_config
            api_any_num_steps = api_config.any_num_steps
            doc_any_num_steps = doc_config.any_num_steps
            api_num_steps = None if api_any_num_steps else api_config.num_steps
            doc_num_steps = None if doc_any_num_steps else doc_config.num_steps
            api_eos = self._api_indexer.word2id('<eos>')
            doc_eos = self._doc_indexer.word2id('<eos>')

            n_examples = 0
            vals = [list() for _ in range(11)]

            def post_precoss(vals):
                if api_any_num_steps:
                    max_len = max(vals[2]) # len(max(vals[0], key=len))
                    vals[0] = unify_len_for_list_of_list(
                        vals[0], pad_value=api_eos, max_len=max_len)
                    vals[1] = unify_len_for_list_of_list(
                        vals[1], pad_value=0., max_len=max_len)
                if doc_any_num_steps:
                    offset = 5
                    max_len = max(vals[offset + 2]) # len(max(vals[offset + 0], key=len))
                    vals[offset + 0] = unify_len_for_list_of_list(
                        vals[offset + 0], pad_value=doc_eos, max_len=max_len)
                    vals[offset + 1] = unify_len_for_list_of_list(
                        vals[offset + 1], pad_value=0., max_len=max_len)
                return vals

            for fn in data_list:
                with codecs.open(fn, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        (api_idxs, api_weights, api_seq_len, tokens,
                         doc_idxs, doc_weights, doc_seq_len, words, cid) = \
                            self._parse_line(line, api_num_steps, doc_num_steps)

                        if ((not api_any_num_steps) and
                                api_seq_len > api_num_steps):
                            continue
                        if ((not doc_any_num_steps) and
                                doc_seq_len > doc_num_steps):
                            continue

                        # Currently, no need to return tokens.
                        tokens = []
                        api_vocab_cnts, doc_vocab_cnts = None, None
                        cur_vals = [api_idxs, api_weights, api_seq_len, tokens,
                                    api_vocab_cnts,
                                    doc_idxs, doc_weights, doc_seq_len, words,
                                    doc_vocab_cnts,
                                    cid]
                        for l, v in zip(vals, cur_vals):
                            l.append(v)

                        if (not batch_all) and len(vals[0]) == batch_size:
                            vals = post_precoss(vals)
                            yield vals
                            vals = [list() for _ in range(11)]
                            n_examples += batch_size

            n_remaining = len(vals[0])
            if n_remaining != 0:
                logging.info('Batching remaining {} examples.'
                             .format(n_remaining))
                n_examples += n_remaining
                vals = post_precoss(vals)
                yield vals

            if batch_all:
                n_batch = int(n_examples > 0)
            else:
                n_batch = ceil(n_examples / batch_size)

            logging.info('Read {} examples ({} batches) in this epoch.'
                         .format(n_examples, n_batch))

    def _to_pickle(self, f, pf,
                   api_weight_idx=1, doc_weight_idx=6, remove_weight=True):
        """Convert to pickle and return original data also."""
        logging.info('Converting {} to pickle'.format(f))
        data = list(self._read_one_epoch([f], do_pickle=False))
        orig_data = data
        if remove_weight:
            orig_data = deepcopy(data)
            for i in range(len(data)):
                data[i][api_weight_idx] = None
                data[i][doc_weight_idx] = None
        with open(pf, 'wb') as pfout:
            pickle.dump(data, pfout)
        return orig_data

    def _from_pickle(self, f, pf,
                     api_idx_idx=0, api_weight_idx=1, api_seq_len_idx=2,
                     api_vocab_cnts_idx=4,
                     doc_idx_idx=5, doc_weight_idx=6, doc_seq_len_idx=7,
                     doc_vocab_cnts_idx=9,
                     recover_weight=True):
        config = self.config
        logging.info('Reading pickle from {}'.format(pf))
        with open(pf, 'rb') as pfin:
            data = pickle.load(pfin)

        def ids2cnt(ids, vocab_size, pad_idx, dtype=np.float32):
            batch_size = len(ids)
            ret = np.zeros([batch_size, vocab_size], dtype=dtype)
            for b in range(batch_size):
                v, c = np.unique(ids[b], return_counts=True)
                ret[b][v] = c
            # Set the pad cnt to 0.
            ret[:, pad_idx] = 0
            return ret

        for i in range(len(data)):
            data[i][api_vocab_cnts_idx] = ids2cnt(
                data[i][api_idx_idx], config.api_config.vocab_size,
                self._api_indexer.word2id(self._api_indexer.EOS))

            data[i][doc_vocab_cnts_idx] = ids2cnt(
                data[i][doc_idx_idx], config.doc_config.vocab_size,
                self._doc_indexer.word2id(self._doc_indexer.EOS))
            # for j in range(11):
            #     print(j, "*"*12)
            #     print(data[i][j])

        show_doc_statitcs = False
        if show_doc_statitcs:
            api_lens = []
            doc_lens = []
            for batch_data in data:
                print(batch_data[api_seq_len_idx])
                api_lens.extend(batch_data[api_seq_len_idx])
                doc_lens.extend(batch_data[doc_seq_len_idx])
            print(np.mean(api_lens), np.median(api_lens))
            print(np.mean(doc_lens), np.median(doc_lens))

        if recover_weight:
            def _get_weights(batch_data, idx_idx, seq_len_idx,
                             indexer, use_idf):
                weights = []
                idxs = batch_data[idx_idx]
                seq_lens = batch_data[seq_len_idx]
                num_steps = len(idxs[0])
                if use_idf:
                    for i in range(len(idxs)):
                        weight = indexer.ids2weights(idxs[i][:seq_lens[i]])
                        weight += [0.] * (num_steps - len(weight))
                        weights.append(weight)
                else:
                    for i in range(len(idxs)):
                        weight = [1.] * seq_lens[i]
                        weight += [0.] * (num_steps - len(weight))
                        weights.append(weight)
                return weights

            for b in range(len(data)):
                # Recover api weight.
                data[b][api_weight_idx] = _get_weights(
                    data[b], api_idx_idx, api_seq_len_idx,
                    self._api_indexer, config.api_config.use_idf)
                # Recover doc weight.
                data[b][doc_weight_idx] = _get_weights(
                    data[b], doc_idx_idx, doc_seq_len_idx,
                    self._doc_indexer, config.doc_config.use_idf)
        return data

    def make_iterators(self, num_repeat=2 ** 31):
        for epoch in range(num_repeat):
            for batch in self._read_one_epoch():
                yield epoch, batch

    def __iter__(self):
        return self._read_one_epoch()

    def read_ref(self, ref_path, api_path='', read_all=False):
        """Read codenn's ref data.

        If `read_all` is True, then also read:
        1) all refs from `ref_path`;
        2) api from `api_path`.
        """
        config = self.config
        doc_config = config.doc_config
        doc_any_num_steps = doc_config.any_num_steps
        doc_num_steps = None if doc_any_num_steps else doc_config.num_steps
        doc_eos = self._doc_indexer.word2id('<eos>')

        class Ref(object):
            def __init__(self):
                # For code retrieval.
                self.cids = []
                self.idxs = []
                self.weights = []
                self.seq_lens = []

                # For code summary.
                self.reader = []
                self.ref_dict = collections.defaultdict(list)

        ref = Ref()

        if read_all:
            logging.info('Reading apis from {} to evaluate summary.'
                         .format(api_path))
            api_reader = self._read_one_epoch(data_list=[api_path],
                                              batch_all=True)
            ref.reader = list(api_reader)

        logging.info('Reading reference docs from {}.'.format(ref_path))

        with codecs.open(ref_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cid, doc = line.split('\t')

                if read_all:
                    if config.lower_doc_idx:
                        doc = doc.lower()
                    ref.ref_dict[cid].append(doc.strip())
                    continue

                if cid in ref.cids:
                    continue

                ref.cids.append(cid)

                add_sos, add_eos = False, True
                words = text2words(doc, lower=config.lower_doc_idx)
                doc_idxs = self._doc_indexer.text2ids(
                    words, add_sos=add_sos, add_eos=add_eos,
                    num_steps=doc_num_steps)
                ref.idxs.append(doc_idxs)

                doc_seq_len = len(words) + add_sos + add_eos
                ref.seq_lens.append(doc_seq_len)

                if doc_config.use_idf:
                    doc_weights = self._doc_indexer.ids2weights(doc_idxs)
                else:
                    doc_weights = [1.] * doc_seq_len
                if doc_num_steps is not None:
                    doc_weights += [0.] * (doc_num_steps - len(doc_weights))

                ref.weights.append(doc_weights)

        if read_all:
            logging.info('Read {} refs in this dataset.'
                         .format(len(ref.ref_dict)))
        else:
            max_len = len(max(ref.idxs, key=len))
            ref.idxs = unify_len_for_list_of_list(
                ref.idxs, pad_value=doc_eos, max_len=max_len)
            ref.weights = unify_len_for_list_of_list(
                ref.weights, pad_value=0., max_len=max_len)

            logging.info('Read {} refs in this dataset.'.format(len(ref.cids)))

        return ref
