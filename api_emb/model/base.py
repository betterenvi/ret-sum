import collections
import time

import numpy as np
import tensorflow as tf

from tensorflow.contrib import distributions
from tensorflow.contrib.bayesflow import stochastic_tensor as st
from tensorflow.contrib.layers import fully_connected as fc_layer

from common.model import Model as AbstractModel
from common.op import kl_divergence


class BaseSeqModel(AbstractModel):
    """BaseSeqModel."""

    def _build_flow(self):
        self._build_inputs()
        self._build_embedding()
        self._build_encoder()
        self._build_latent()
        self._build_decoder()
        self._build_loss()

    def _build_inputs(self):
        config = self.config

        b = config.batch_size
        v = config.vocab_size
        t = config.num_steps

        if config.any_batch_size:
            b = None

        if config.any_num_steps:
            t = None

        self.bow_inputs = tf.placeholder(tf.float32, [b, v], name="bow_inputs")
        self.idx_inputs = tf.placeholder(tf.int32, [b, t], name='idx_inputs')
        self.weights = tf.placeholder(tf.float32, [b, t], name='weights')
        self.seq_lens = tf.placeholder(tf.int32, [b], name='seq_lens')

        bt_shape = tf.shape(self.idx_inputs)
        self.seq_mask = tf.sequence_mask(self.seq_lens,
                                         maxlen=bt_shape[1],
                                         dtype=tf.float32,
                                         name='seq_mask')
        self.seq_mask_3d = tf.expand_dims(self.seq_mask, axis=2)
        self.weights_3d = tf.expand_dims(self.weights, axis=2)

        b_idxs = tf.range(bt_shape[0])
        b_idxs = tf.expand_dims(b_idxs, axis=1)
        b_idxs = tf.tile(b_idxs, [1, bt_shape[1]])
        b_idxs = tf.reshape(b_idxs, [-1])
        self.gather_idxs = tf.stack(
            [b_idxs, tf.reshape(self.idx_inputs, [-1])],
            axis=1)

        # For sigoid activation.
        self.idx_in_vocab = tf.cast(tf.greater(self.bow_inputs, 0.), tf.float32)
        self.idx_nin_vocab = tf.ones_like(self.idx_in_vocab) - self.idx_in_vocab

        self.vocab_weights = tf.constant(self.kwargs['vocab_weights'],
                                         dtype=tf.float32)

    def _build_embedding(self):
        config = self.config
        v, e = config.vocab_size, config.embedding_size
        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable('embedding',
                                             shape=[v, e], dtype=tf.float32)
            self.hinputs = tf.nn.embedding_lookup(self.embedding,
                                                  self.idx_inputs)

    def _build_encoder(self):
        raise NotImplementedError()

    def _build_latent(self):
        config = self.config

        hinputs = self.encoder_logits

        self.latent_mu = fc_layer(hinputs, config.latent_size,
                                  activation_fn=None)

        # Use softplus or exp as activation function.
        self.latent_sd = tf.nn.softplus(fc_layer(
            hinputs, config.latent_size, activation_fn=None))

        if not config.train_encoder:
            self.latent_mu = tf.stop_gradient(self.latent_mu)
            self.latent_sd = tf.stop_gradient(self.latent_sd)

        with st.value_type(st.SampleValue()):
            self.latent_posterior_dist = distributions.Normal(
                self.latent_mu,
                self.latent_sd)

            if config.use_variational:
                self.latent = \
                    st.StochasticTensor(self.latent_posterior_dist)
            else:
                self.latent = self.latent_mu

            if config.use_posterior_mu:
                prior_mu = self.latent_mu
            else:
                prior_mu = tf.zeros_like(self.latent_mu, dtype=tf.float32)

            prior_sd = tf.ones_like(self.latent_sd, dtype=tf.float32) * \
                config.latent_prior_sd

            self.latent_prior_dist = distributions.Normal(prior_mu, prior_sd)
            self.latent_prior = st.StochasticTensor(self.latent_prior_dist)

    def _build_decoder(self):
        """Generator Network, i.e. p(X|h)."""
        config = self.config

        # If mode is not 'train', feed mu to decoder directly.
        hinputs = self.latent if self.is_train else self.latent_mu

        self.decoder_hiddens = []
        with tf.variable_scope('decoder'):
            for lid in range(config.num_decoder_layers):
                hinputs = fc_layer(hinputs,
                                   config.decoder_hidden_size,
                                   activation_fn=tf.nn.tanh)
                self.decoder_hiddens.append(hinputs)
            self.logits = fc_layer(hinputs, config.vocab_size,
                                   activation_fn=None)
            self.p_x_hat = tf.nn.softmax(self.logits, dim=-1, name='p_x_hat')
            # self.p_x_hat = tf.sigmoid(self.logits, name='p_x_hat')

    def _build_kl_div(self):
        config = self.config
        # KL divergence as a regularizer.
        if config.use_variational:
            self.kl_div = kl_divergence(self.latent.distribution,
                                        self.latent_prior.distribution,
                                        average_across_latent_dim=False,
                                        average_across_batch=True)
        else:
            self.kl_div = tf.constant(0., dtype=tf.float32)

    def _build_log_prob(self):
        config = self.config

        eps = tf.constant(1e-12, dtype=tf.float32)
        prob = tf.gather_nd(self.p_x_hat, self.gather_idxs)
        b_shape = tf.shape(self.idx_inputs)[0]
        prob = tf.reshape(prob, [b_shape, -1])
        log_prob = tf.log(prob + eps)
        self.log_prob = log_prob    # [b, t]

    def _build_loss(self):
        config = self.config

        eps = tf.constant(1e-12, dtype=tf.float32)

        self._build_log_prob()
        log_prob = self.log_prob
        # Calc loss.
        log_prob = log_prob * self.weights
        if config.average_across_timesteps:
            size = tf.reduce_sum(self.weights)
            log_prob = tf.reduce_sum(log_prob)
            log_prob = log_prob / (size + eps)
        else:
            log_prob = tf.reduce_sum(log_prob, axis=1)
            log_prob = tf.reduce_mean(log_prob)
        self.neg_log_likelihood = -log_prob

        self._build_kl_div()

        self.loss = tf.constant(0., dtype=tf.float32)
        if config.kl_div_wht > 0.:
            self.loss += self.kl_div * config.kl_div_wht
        if config.kl_div_wht < 1.:
            self.loss += self.neg_log_likelihood * (1 - config.kl_div_wht)

    def _build_targets_and_metrics(self):
        self.perf_key = 'loss'
        self.perf_large_is_better = False
        self.metric_keys = ['kl_div', 'loss']
        for k, v in zip(self.metric_keys, [self.kl_div, self.loss]):
            self.run_targets[k] = v

        if self.is_train:
            self.run_targets['train'] = self.train_op
        else:
            self.run_targets['latent_mu'] = self.latent_mu
            self.run_targets['latent_sd'] = self.latent_sd

    def _build_reader(self):
        self.reader = None

    def run_batch(self, sess, batch_data, show=True, batch_data_offset=0):
        offset = batch_data_offset
        fd = {
            self.idx_inputs: batch_data[offset + 0],
            self.weights: batch_data[offset + 1],
            self.seq_lens: batch_data[offset + 2],
            self.bow_inputs: batch_data[offset + 4],
        }
        targets = sess.run(self.run_targets, feed_dict=fd)
        return targets

    def run_epoch(self, sess, reader=None, print_info=False,
                  global_step=None, batch_data_offset=0, max_batch=np.inf):
        """Run one epoch.

        Args:
            reader: a iterable object, such a reader or a list of batch.
        """
        config = self.config
        reader = reader or self.reader

        n_batch_metrics = collections.defaultdict(float)
        n_batch_num_examples = 0

        prev_time = time.time()
        for batch, batch_data in enumerate(reader):
            if batch >= max_batch:
                break
            every_n = (batch + 1) % config.print_every_n_batch == 0
            show = print_info and every_n
            batch_targets = self.run_batch(sess, batch_data, show=show,
                                           batch_data_offset=batch_data_offset)

            # Update accumulated and n batch targets.
            cur_batch_size = len(batch_data[0])
            self.num_examples += cur_batch_size
            n_batch_num_examples += cur_batch_size
            for k in self.metric_keys:
                self.acc_metrics[k] += batch_targets[k] * cur_batch_size
                n_batch_metrics[k] += batch_targets[k] * cur_batch_size

            # Show range of batchs, if print_info and every_n.
            if show:
                b_str = 'Epoch {} batch {} - {}: '.format(
                    self.num_epoch,
                    batch - config.print_every_n_batch + 1, batch)
                s = self.metrics2str(self.metric_keys, n_batch_metrics,
                                     n_batch_num_examples)
                now_time = time.time()
                time_cost = now_time - prev_time
                prev_time = now_time
                s = b_str + s + ', time = {:.3f}'.format(time_cost)
                self.logger.info(s)

                n_batch_metrics = collections.defaultdict(float)
                n_batch_num_examples = 0

        self.num_epoch += 1

        self._finish_epoch(sess)

        if self.num_examples == 0:
            return 0.
        else:
            return self.acc_metrics['loss'] / self.num_examples

    def _mask(self, inputs, by='weight'):
        ndims = inputs.shape.ndims

        if ndims == 2:
            m = self.weights if by == 'weight' else self.seq_mask
            return m * inputs
        elif ndims == 3:
            m = self.weights_3d if by == 'weight' else self.seq_mask_3d
            return m * inputs
        else:
            raise NotImplementedError('Not Implemented.')

    def show_mu_sd(self, arr, axis=0, msg='', size_to_show=None):
        mu = np.mean(arr, axis=axis)
        sd = np.std(arr, axis=axis)

        if size_to_show is not None:
            mu = mu[:size_to_show]
            sd = sd[:size_to_show]

        def list2str(lst):
            return ' '.join(['{:.3f}'.format(v) for v in lst])

        self.logger.info(' '.join([msg, 'mu', list2str(mu)]))
        self.logger.info(' '.join([msg, 'sd', list2str(sd)]))
