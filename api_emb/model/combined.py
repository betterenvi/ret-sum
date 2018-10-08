"""Model of learning API embedding."""
import logging
import os

import numpy as np
import tensorflow as tf

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from tensorflow.contrib import distributions

from common.model import Model
from common.op import cosine_sim
from common.op import kl_divergence
from common.perf import Metric
from common.perf import Perf
from common.util import mean_confidence_interval
from common.util import sort_lists
from common.util import split_arr_by_diag
from common.util import walkdir

from ..index import Indexer
from ..metric.bleu import get_bleu
from ..metric.meteor import get_meteor
from ..reader import Reader


class ApiDocVAE(Model):
    """ApiDocVAE."""

    @classmethod
    def static_initialize(cls, config):
        api_config = config.api_config
        doc_config = config.doc_config
        cls.api_indexer = Indexer(config.api_idx_file,
                                  min_cnt=config.min_api_cnt, name='api',
                                  eos_wht=api_config.eos_wht)
        cls.doc_indexer = Indexer(config.doc_idx_file,
                                  min_cnt=config.min_doc_cnt, name='doc',
                                  eos_wht=doc_config.eos_wht)

        api_config.vocab_size = cls.api_indexer.num_words
        logging.info('Api index built with {} words.'
                     .format(cls.api_indexer.num_words))
        doc_config.vocab_size = cls.doc_indexer.num_words
        logging.info('Doc index built with {} words.'
                     .format(cls.doc_indexer.num_words))

    @classmethod
    def get_reader(cls, data_dir, data_ext, config, exact_match=False):
        data_list = walkdir(data_dir, extension=data_ext,
                            exact_match=exact_match)
        reader = Reader(config, data_list, cls.api_indexer, cls.doc_indexer)
        return reader

    def _build_flow(self):
        config = self.config

        with tf.variable_scope('api'):
            api_vocab_size = config.api_config.vocab_size
            api_vocab_weights = np.array(
                self.api_indexer.ids2weights(range(api_vocab_size)))
            api_model = self.kwargs['api_model']
            self.api_vae = api_model(config.api_config,
                                     mode=self.mode, reuse=self.reuse,
                                     name='api', vs='api_vs',
                                     task=config.task,
                                     indexer=self.doc_indexer,
                                     ad_latent=None,
                                     vocab_weights=api_vocab_weights,
                                     **self.kwargs)

        with tf.variable_scope('doc'):
            if not self.is_train:
                ad_latent = self.api_vae.latent_mu
            else:
                ad_latent = None

            doc_vocab_size = config.doc_config.vocab_size
            doc_vocab_weights = np.array(
                self.doc_indexer.ids2weights(range(doc_vocab_size)))
            doc_model = self.kwargs['doc_model']
            self.doc_vae = doc_model(config.doc_config,
                                     mode=self.mode, reuse=self.reuse,
                                     name='doc', vs='doc_vs',
                                     task=config.task,
                                     indexer=self.doc_indexer,
                                     ad_latent=ad_latent,
                                     vocab_weights=doc_vocab_weights,
                                     **self.kwargs)
        # Loss.
        if config.use_variational:
            self.ad_loss = kl_divergence(
                self.api_vae.latent.distribution,
                self.doc_vae.latent.distribution,
                average_across_latent_dim=False,
                average_across_batch=True)
            self.da_loss = kl_divergence(
                self.doc_vae.latent.distribution,
                self.api_vae.latent.distribution,
                average_across_latent_dim=False,
                average_across_batch=True)
            if config.sim_loss_option == 'klm':
                self.latent_mu = (self.api_vae.latent_mu +
                                  self.doc_vae.latent_mu) / 2
                if config.use_fix_sd_in_klm:
                    self.latent_sd = \
                        tf.ones_like(self.latent_mu, dtype=tf.float32) * \
                        config.klm_latent_prior_sd
                else:
                    var = (self.api_vae.latent_sd ** 2 +
                           self.doc_vae.latent_sd ** 2) / 4
                    self.latent_sd = tf.sqrt(var)
                self.latent_dist = distributions.Normal(
                    self.latent_mu, self.latent_sd)
                self.ad_sim_loss = kl_divergence(
                    self.api_vae.latent.distribution,
                    self.latent_dist,
                    average_across_latent_dim=False,
                    average_across_batch=True)
                self.da_sim_loss = kl_divergence(
                    self.doc_vae.latent.distribution,
                    self.latent_dist,
                    average_across_latent_dim=False,
                    average_across_batch=True)
                self.logger.info('Use klm div as sim loss.')
            elif config.sim_loss_option == 'kl':
                self.ad_sim_loss = self.ad_loss
                self.da_sim_loss = self.da_loss
                self.logger.info('Use kl div as sim loss.')
            else:
                raise NotImplementedError()
        else:
            d = self.api_vae.latent_mu - self.doc_vae.latent_mu
            l = tf.reduce_mean(tf.reduce_sum(d ** 2, axis=1), axis=0)
            self.ad_sim_loss = l
            self.da_sim_loss = l
            self.logger.info('Use euclidean as sim loss.')

        self.sim_loss = (self.ad_sim_loss + self.da_sim_loss) / 2

        self.loss = tf.constant(0., dtype=tf.float32)

        sim_wht = 1 - config.api_loss_wht - config.doc_loss_wht
        if sim_wht > 0.:
            self.loss += self.sim_loss * sim_wht
        if config.api_loss_wht > 0.:
            self.loss += config.api_loss_wht * self.api_vae.loss
        if config.doc_loss_wht > 0.:
            self.loss += config.doc_loss_wht * self.doc_vae.loss

    def _build_targets_and_metrics(self):
        config = self.config
        self.perf_key = 'loss'
        self.perf_large_is_better = False

        keys = []
        vals = []

        keys += ['api_' + k for k in self.api_vae.metric_keys]
        vals += [self.api_vae.run_targets[k] for k in self.api_vae.metric_keys]
        keys += ['doc_' + k for k in self.doc_vae.metric_keys]
        vals += [self.doc_vae.run_targets[k] for k in self.doc_vae.metric_keys]

        if config.sim_loss_option == 'klm':
            keys += ['ad_sim', 'ad', 'da_sim', 'da', 'loss']
            vals += [self.ad_sim_loss, self.ad_loss,
                     self.da_sim_loss, self.da_loss, self.loss]
        else:
            keys += ['ad_sim', 'da_sim', 'loss']
            vals += [self.ad_sim_loss, self.da_sim_loss, self.loss]

        self.metric_keys = keys
        for k, v in zip(keys, vals):
            self.run_targets[k] = v

        if self.is_train:
            self.run_targets['train'] = self.train_op

    def _build_restore_prefixs(self):
        config = self.config
        self.restore_prefixs = []

        if config.restore_api:
            self.restore_prefixs.append(self.api_vae.var_scope.name)
        if config.restore_doc:
            self.restore_prefixs.append(self.doc_vae.var_scope.name)

    def _build_reader(self):
        self._build_data_list()
        self.reader = Reader(self.config, self.data_list,
                             self.api_indexer, self.doc_indexer)

    def run_batch(self, sess, batch_data, show=True, batch_data_offset=0):
        api_offset = batch_data_offset
        doc_offset = batch_data_offset + 5
        fd = {
            self.api_vae.idx_inputs: batch_data[api_offset + 0],
            self.api_vae.weights: batch_data[api_offset + 1],
            self.api_vae.seq_lens: batch_data[api_offset + 2],
            self.api_vae.bow_inputs: batch_data[api_offset + 4],
            self.doc_vae.idx_inputs: batch_data[doc_offset + 0],
            self.doc_vae.weights: batch_data[doc_offset + 1],
            self.doc_vae.seq_lens: batch_data[doc_offset + 2],
            self.doc_vae.bow_inputs: batch_data[doc_offset + 4]
        }
        targets = sess.run(self.run_targets, feed_dict=fd)
        return targets

    def get_sims(self, x, y, method='cos'):
        funcs = {
            'cos': cosine_similarity,
            'euc': euclidean_distances,
        }
        func = funcs[method]
        return func(x, y)

    def calc_mrr(self, gold_sims, other_sims,
                 num_eval_times=20,
                 num_candidates=50,
                 confidence=0.95):
        config = self.config

        mrrs = []
        gold_sims = gold_sims.reshape([-1, 1])
        n_examples, n_others = other_sims.shape

        # In shape comments, denote n_examples by 'n', num_candidates by 'c'.

        rows = np.arange(n_examples).reshape(-1, 1)
        rows = np.tile(rows, [1, num_candidates - 1])
        rows = np.reshape(rows, [-1])                           # [n * (c - 1)]
        columns = np.arange(n_others)
        for t in range(num_eval_times):
            cmp_idxs = [np.random.choice(columns,
                                         num_candidates - 1,
                                         replace=False)
                        for _ in range(n_examples)]
            cmp_idxs = np.reshape(cmp_idxs, [-1])
            cmp_sims = other_sims[rows, cmp_idxs]
            cmp_sims = cmp_sims.reshape(n_examples, -1)         # [n, c - 1]
            # Then extract gold api_repr's rank.
            rank = (cmp_sims > gold_sims).sum(axis=1) + 1       # [n]
            mrr = np.mean(1. / rank)
            mrrs.append(mrr)

        m, h = mean_confidence_interval(mrrs, confidence=confidence)
        mrr = Metric(m, h, confidence=confidence)

        self.logger.info('In {} candidates, eval {} times, by {} confidence: MRR {}'
                         .format(num_candidates, num_eval_times, confidence, mrr))
        return mrr

    def evaluate_codenn(self, sess, reader=None,
                        print_info=False, batch_data_offset=0,
                        data_mode='test'):
        """Evaluate model in codenn way.

        Args:
        """
        config = self.config

        # Only test on test data according to Codenn.
        if data_mode != 'test':
            self.logger.info('Skipping evaluation. Validation on DEV set will be performed together with EVAL set. '
                             'Please see the result at test stage.')
            return Metric(1., 0.)

        if not hasattr(self, 'perf'):
            self.perf = Perf(name='mrr', large_is_better=True)
        if not hasattr(self, 'dev_ref'):
            self.dev_ref = self.reader.read_ref(config.codenn_dev_ref)
        if not hasattr(self, 'eval_ref'):
            self.eval_ref = self.reader.read_ref(config.codenn_eval_ref)

        offset = batch_data_offset

        api_cids = []
        for batch, batch_data in enumerate(reader):
            api_cids.extend(batch_data[offset + 10])

        if (len(set(api_cids) & set(self.dev_ref.cids)) == 0 or
                len(set(api_cids) & set(self.eval_ref.cids)) == 0):
            return None

        self.logger.info('----- evaluating in codenn way... -----')

        api_reprs = []
        api_offset = batch_data_offset
        for batch, batch_data in enumerate(reader):
            fd = {
                self.api_vae.idx_inputs: batch_data[api_offset + 0],
                self.api_vae.weights: batch_data[api_offset + 1],
                self.api_vae.seq_lens: batch_data[api_offset + 2],
                self.api_vae.bow_inputs: batch_data[api_offset + 4],
            }
            api_repr = sess.run(self.api_vae.latent_mu, feed_dict=fd)
            api_reprs.extend(api_repr)

        def _get_ref_reprs(ref):
            fd = {
                self.doc_vae.idx_inputs: ref.idxs,
                self.doc_vae.weights: ref.weights,
                self.doc_vae.seq_lens: ref.seq_lens
            }
            ref_reprs = sess.run(self.doc_vae.latent_mu, feed_dict=fd)
            return ref_reprs

        def _run_eval_codenn(ref_cids, ref_reprs, api_cids, api_reprs):
            ref_cids, ref_reprs = sort_lists([ref_cids, ref_reprs],
                                             ref_cids, reverse=True)

            sort_key = [k if k in ref_cids else '' for k in api_cids]
            api_cids, api_reprs = sort_lists([api_cids, api_reprs],
                                             sort_key, reverse=True)

            assert ref_cids == api_cids[:len(ref_cids)]

            all_sims = cosine_similarity(ref_reprs, api_reprs)
            gold_sims, other_sims = split_arr_by_diag(all_sims)
            mrr = self.calc_mrr(gold_sims, other_sims,
                                num_eval_times=config.num_eval_times,
                                num_candidates=config.num_candidates,
                                confidence=config.confidence)
            return mrr

        self.logger.info('  --  evaluating on dev ref  --  ')
        dev_ref = self.dev_ref
        dev_ref_reprs = _get_ref_reprs(dev_ref)
        valid_mrr = _run_eval_codenn(dev_ref.cids, dev_ref_reprs,
                                     api_cids, api_reprs)
        self.perf.update(valid_mrr, mode='valid', log=True)

        if config.test_when_valid:
            self.logger.info('  --  evaluating on eval ref --  ')
            eval_ref = self.eval_ref
            eval_ref_reprs = _get_ref_reprs(eval_ref)
            eval_mrr = _run_eval_codenn(eval_ref.cids, eval_ref_reprs,
                                        api_cids, api_reprs)
            self.perf.update(eval_mrr, mode='test', log=True)
            self.logger.info("-------- Result below --------")
            self.perf.show()
            self.logger.info("-------- Result above --------")

        return valid_mrr

    def evaluate_summary(self, sess, reader=None,
                         print_info=False, batch_data_offset=0,
                         data_mode='valid'):
        """Evaluate code summary.

        Args:
            reader: this arg is not used.
                    The data to be evaluated is read inside this method.

        Return BLEU score.
        """
        config = self.config

        if data_mode == 'train':
            return 0.

        if not hasattr(self, 'bleu_perf'):
            self.bleu_perf = Perf(name='BLEU', large_is_better=True)

        api_vae = self.api_vae
        doc_vae = self.doc_vae

        is_valid = data_mode == 'valid'
        is_test = data_mode == 'test'

        # Read data to be evaluated.
        def _prepare_ref_texts(ref):
            ref_texts = []
            ref_sum_cids = []
            for _, batch_data in enumerate(ref.reader):
                ref_sum_cids.extend(batch_data[10])
                ref_texts.extend([ref.ref_dict[cid] for cid in batch_data[10]])
            ref.ref_texts = ref_texts
            ref.ref_sum_cids = ref_sum_cids
        if is_valid and (not hasattr(self, 'sum_dev_ref')):
            self.sum_dev_ref = self.reader.read_ref(
                config.codenn_sum_dev_ref,
                api_path=config.codenn_sum_dev_api, read_all=True)
            _prepare_ref_texts(self.sum_dev_ref)
        if is_test and (not hasattr(self, 'sum_eval_ref')):
            self.sum_eval_ref = self.reader.read_ref(
                config.codenn_sum_eval_ref,
                api_path=config.codenn_sum_eval_api, read_all=True)
            _prepare_ref_texts(self.sum_eval_ref)

        def _evaluate(ref, data_mode='valid'):
            reader = ref.reader
            offset = 0

            predicts = []
            lens = []

            api_offset = batch_data_offset
            doc_offset = batch_data_offset + 5
            for batch, batch_data in enumerate(reader):
                fd = {
                    api_vae.idx_inputs: batch_data[api_offset + 0],
                    api_vae.weights: batch_data[api_offset + 1],
                    api_vae.seq_lens: batch_data[api_offset + 2],
                    api_vae.bow_inputs: batch_data[api_offset + 4],
                    # For diagnosing latent distribution.
                    doc_vae.idx_inputs: batch_data[doc_offset + 0],
                    doc_vae.weights: batch_data[doc_offset + 1],
                    doc_vae.seq_lens: batch_data[doc_offset + 2],
                    doc_vae.bow_inputs: batch_data[doc_offset + 4],
                }
                (api_latent_mu, api_latent_sd,
                    doc_latent_mu, doc_latent_sd, cur_predicts, cur_lens) = \
                    sess.run([api_vae.latent_mu,
                              api_vae.latent_sd,
                              doc_vae.latent_mu,
                              doc_vae.latent_sd,
                              doc_vae.predicts,
                              doc_vae.final_sequence_lengths],
                             feed_dict=fd)

                # Only save the best one.
                predicts.extend(cur_predicts[:, :, 0])
                lens.extend(cur_lens[:, 0])

            # Cut redundant ids. Cut <eos>, too.
            _, pred_texts = doc_vae.ids2texts(predicts, lens)

            # Eval BLEU-4.
            import os
            fn = 'dev-sum.txt' if is_valid else 'eval-sum.txt'
            fn = os.path.join(config.model_path, fn)
            with open(fn, 'w') as fout:
                for cid, tx in zip(ref.ref_sum_cids, pred_texts):
                    fout.write('{}\t{}\n'.format(cid, tx))

            bleu = get_bleu(pred_texts, ref.ref_texts)
            self.logger.info('BLEU: ' + str(bleu[0]))
            self.bleu_perf.update(bleu[0], mode=data_mode, log=True)
            if data_mode == 'test':
                self.bleu_perf.show()

            # Eval METEOR.
            numref = len(ref.ref_texts[0])
            meteor = get_meteor(pred_texts,
                                ref.ref_texts, numref=numref,
                                meteor_jar_path=config.meteor_jar_path,
                                rewrite_meteor_refs=False,
                                pre_tokenize=True, norm=False)
            self.logger.info('METEOR: {}'.format(meteor))
            score = Metric((meteor, bleu[0]), h=None, large_is_better=True)
            return score

        if is_valid:
            score = _evaluate(self.sum_dev_ref, data_mode='valid')

        if is_test:
            score = _evaluate(self.sum_eval_ref, data_mode='test')

        return score

    def on_valid_updated(self, sess):
        config = self.config

        if config.task == 'sum':
            for f in ['dev-sum.txt', 'eval-sum.txt']:
                fn = os.path.join(config.model_path, f)
                os.system('cp {} {}.best'.format(fn, fn))

    def evaluate(self, sess, reader=None,
                 print_info=False, batch_data_offset=0, data_mode='valid'):
        config = self.config
        reader = reader or self.reader

        if config.task == 'ret':    # Code retrieval.
            return self.evaluate_codenn(sess, reader=reader,
                                 print_info=print_info,
                                 batch_data_offset=batch_data_offset,
                                 data_mode=data_mode)
        elif config.task == 'sum':      # Code summary.
            return self.evaluate_summary(
                sess, reader=reader,
                print_info=print_info,
                batch_data_offset=batch_data_offset,
                data_mode=data_mode)
        else:
            raise NotImplementedError()

    def run_epoch(self, sess, reader=None, print_info=False,
                  global_step=None, batch_data_offset=0, max_batch=np.inf):
        config = self.config

        reader = reader or self.reader

        ret = super(ApiDocVAE, self).run_epoch(
            sess, reader,
            print_info=print_info,
            global_step=global_step,
            batch_data_offset=batch_data_offset,
            max_batch=max_batch)

        return ret
