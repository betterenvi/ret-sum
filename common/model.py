"""Class of base model."""
import collections
import logging
import os
import time

import numpy as np
import tensorflow as tf

from common.op import SoftSaver
from common.util import walkdir


class Model(object):
    """Base model."""
    static_initialized_classes = set()

    @classmethod
    def static_initialize(cls, config):
        pass

    @classmethod
    def get_reader(cls, data_dir, data_ext, config, exact_match=False):
        raise NotImplementedError()

    @classmethod
    def metrics2str(cls, metric_keys, metrics_dict, num_examples):
        s = ', '.join(['{} = {:.2f}'
                       .format(k, metrics_dict[k] / (num_examples + 1e-10))
                       for k in metric_keys])
        return s

    def __init__(self, config, mode='train', reuse=None,
                 name='model', vs='model_vs', **kwargs):
        """Init Model."""
        super(Model, self).__init__()

        c = self.__class__
        if c not in c.static_initialized_classes:
            c.static_initialize(config)
            c.static_initialized_classes.add(c)

        self.kwargs = kwargs

        self.logger = logging.getLogger(name)
        self.config = config
        self.mode = mode
        self.name = name
        self.is_train = mode == 'train'
        self.reuse = reuse
        self.vs = vs

        self.num_epoch = 0
        self.num_examples = 0
        self.run_targets = {}
        self.metric_keys = []

        self.perf_key = 'loss'
        self.perf_large_is_better = False

        self.acc_metrics = collections.defaultdict(float)

        self._build_reader()

        initializer = tf.contrib.layers.xavier_initializer()
        with tf.name_scope(name), tf.variable_scope(
                vs, reuse=self.reuse, initializer=initializer) as var_scope:
            self.var_scope = var_scope
            self._build_flow()
            if self.is_train:
                self._build_train_op()
        self.logger.info('Model {} built.'.format(self.__class__.__name__))
        self._build_targets_and_metrics()
        self._build_restore_prefixs()

        # Checking metrics.
        assert self.perf_key in self.metric_keys
        for k in self.metric_keys:
            assert k in self.run_targets

    def _build_train_op(self):
        config = self.config

        self.lr = tf.Variable(config.lr, trainable=False)
        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
        self.assign_lr_op = tf.assign(self.lr, self.lr_placeholder)

        optimizers = {
            'gd': tf.train.GradientDescentOptimizer,
            'adam': tf.train.AdamOptimizer,
            'rms': tf.train.RMSPropOptimizer,
        }
        optimizer_name = config.optimizer.lower()
        self.optimizer = optimizers[optimizer_name](learning_rate=self.lr)
        self.logger.info('Using {} optimizer.'.format(config.optimizer))

        gvs = self.optimizer.compute_gradients(self.loss)
        if config.clip_grad > 0:
            new_gvs = []
            for grad, var in gvs:
                if (var is not None) and (grad is not None):
                    new_gvs.append((grad, var))
            gvs = [(tf.clip_by_value(grad, -config.clip_grad,
                                     config.clip_grad), var)
                   for grad, var in new_gvs]
        self.train_op = self.optimizer.apply_gradients(gvs)

    def _build_flow(self):
        raise NotImplementedError()

    def _build_targets_and_metrics(self):
        self.run_targets['train'] = self.train_op

    def _build_restore_prefixs(self):
        self.restore_prefixs = [self.var_scope.name]

    def _build_data_list(self):
        config = self.config
        if self.is_train:
            data_list = walkdir(config.train_dir, extension=config.train_ext,
                                exact_match=config.train_exact)
            self.logger.info('Found {} train files.'.format(len(data_list)))
        else:
            data_list = walkdir(config.valid_dir, extension=config.valid_ext,
                                exact_match=config.valid_exact)
            self.logger.info('Found {} valid files.'.format(len(data_list)))
        self.data_list = data_list

    def _build_reader(self):
        self.reader = None
        raise NotImplementedError()

    def run_batch(self, sess, batch_data, show=True, batch_data_offset=0):
        raise NotImplementedError()

    def clear_metric(self):
        self.num_examples = 0
        for k in self.acc_metrics:
            self.acc_metrics[k] = 0

    def print_metric(self):
        if self.num_examples == 0:
            return
        s = self.metrics2str(self.metric_keys, self.acc_metrics,
                             self.num_examples)
        self.logger.info(s)

        perf = self.acc_metrics[self.perf_key] / self.num_examples
        return perf

    def better_perf(self, perf1, perf2):
        if perf2 is None:
            return True
        if perf1 is None:
            return False
        if self.perf_large_is_better:
            return perf1 > perf2
        else:
            return perf1 < perf2

    def get_perf_attr(self):
        return self.perf_key, self.perf_large_is_better

    def infer(self, batch_data):
        raise NotImplementedError()

    def evaluate(self, sess, reader=None,
                 print_info=False, batch_data_offset=0, data_mode='valid'):
        pass

    def on_valid_updated(self, sess):
        pass

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

    def _finish_epoch(self, sess):
        pass

    def init_session(self, sess, model_path=None):
        init_op = [tf.local_variables_initializer(),
                   tf.global_variables_initializer()]
        sess.run(init_op)
        if self.config.continue_run and model_path:
            self.restore_session(sess, model_path)
        self.logger.info('Session initialized.')

    def restore_session(self, sess, model_path, name_converter=None):
        try:
            self.logger.info('Trying to restore from {}'.format(model_path))
            if os.path.isdir(model_path):
                check_point = tf.train.latest_checkpoint(model_path)
            else:
                check_point = model_path
            saver = SoftSaver()
            self.logger.info('Prefixs to restore: {}'
                             .format(str(self.restore_prefixs)))
            saver.restore(sess, check_point, soft=True,
                          name_trans_func=name_converter,
                          prefixs=self.restore_prefixs)
        except Exception as e:
            self.logger.info('Session restore failed due to: {}'.format(e))
        else:
            self.logger.info('Session restored from {}.'.format(check_point))
        finally:
            pass

    def assign_lr(self, sess, lr):
        sess.run([self.assign_lr_op], feed_dict={self.lr_placeholder: lr})
        self.logger.info('Learning rate assigned {}'.format(lr))
