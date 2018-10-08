"""Controller."""
import logging
import os
import time

import tensorflow as tf

from .util import LRAdjust

from .perf import Perf


def do_every(i, every, limit, bias=1):
    t = i + bias
    return t % every == 0 or t == limit


class Controller(object):
    """Controller.
    Model: a model class to train and validate.
    """

    def __init__(self, config, Model, **kwargs):
        """Init Controller."""
        super(Controller, self).__init__()

        self.config = config
        self.Model = Model

        self.kwargs = kwargs

        self.initialized = False
        self.restored = False

        # Create session.
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.gpu_options.allocator_type = 'BFC'
        sess_config.allow_soft_placement = True
        sess = tf.Session(config=sess_config)
        self.sess = sess

        # Set logging level.
        log_levels = {
            'info': logging.INFO,
            'warn': logging.WARNING,
        }

        if config.log_to_file:
            logging.basicConfig(filename=config.log_file)

        logging.basicConfig(level=log_levels[config.log_level.lower()])

    def run(self):
        config = self.config
        if config.mode == 'train':
            self.train()
        elif config.mode == 'valid':
            self.valid()
        elif config.mode == 'infer':
            self.infer()
        else:
            raise NotImplementedError('This mode is not implemented.')

    def train(self):
        """Train model."""
        config = self.config
        sess = self.sess

        if not self.initialized:
            self.initialized = True
            Model = self.Model
            self.train_model = Model(config, mode='train', reuse=None,
                                     **self.kwargs)
            self.valid_model = Model(config, mode='valid', reuse=True,
                                     **self.kwargs)
            self._prepare_lr_adjust()
            restore_path = config.restore_path or config.model_path
            model_dir = os.path.dirname(restore_path)
            self.train_model.init_session(sess, model_dir)

        train_model = self.train_model
        valid_model = self.valid_model
        lr_adjust = self._lr_adjust

        lr = config.lr

        train_reader = train_model.reader
        if config.train_prefetch:
            logging.info('Prefetching train data.')
            train_reader = list(iter(train_reader))

        if config.valid_train_data:
            if config.train_prefetch:
                valid_train_reader = train_reader[:config.max_valid_train_batch]
            else:
                logging.info('Prefetching first {} batches of train data.'
                             .format(config.max_valid_train_batch))
                itr = iter(train_reader)
                valid_train_reader = \
                    [next(itr) for _ in range(config.max_valid_train_batch)]
        else:
            valid_train_reader = []

        valid_reader = valid_model.reader
        if config.valid_prefetch:
            logging.info('Prefetching valid data.')
            valid_reader = list(iter(valid_reader))

        if config.test_when_valid:
            test_reader = self.Model.get_reader(
                data_dir=config.test_dir, data_ext=config.test_ext,
                config=config, exact_match=config.test_exact)
            if config.test_prefetch:
                logging.info('Prefetching test data.')
                test_reader = list(iter(test_reader))
        else:
            test_reader = []

        perf_key, large_is_better = valid_model.get_perf_attr()
        train_batch_perf = Perf(name=perf_key, large_is_better=large_is_better)
        train_eval_perf = Perf(name='eval', large_is_better=True)
        batch_perf = Perf(name=perf_key, large_is_better=large_is_better)
        eval_perf = Perf(name='eval', large_is_better=True)

        # Valids first, in case we restore a best model and then overwrite it.
        def valid_perf():

            def _run_valid_perf(valid_reader, batch_perf, eval_perf,
                                data_mode='valid'):
                logging.info('===== Validation starts =====')
                if config.valid_accord in {'all', 'batch'}:
                    start = time.time()
                    batch_perf.best_valid_updated = False
                    valid_model.clear_metric()
                    valid_model.run_epoch(sess, reader=valid_reader,
                                          print_info=False,
                                          max_batch=config.max_valid_batch)
                    cur_perf = valid_model.print_metric()
                    batch_perf.update(cur_perf, mode='valid', log=True)
                    cost = time.time() - start
                    logging.info('Valid by batch, time: {:.4f}'.format(cost))
                if config.valid_accord in {'all', 'eval'}:
                    start = time.time()
                    eval_perf.best_valid_updated = False
                    cur_eval = valid_model.evaluate(sess, reader=valid_reader,
                                                    print_info=False,
                                                    data_mode=data_mode)
                    eval_perf.update(cur_eval, mode='valid', log=True)
                    cost = time.time() - start
                    logging.info('Valid by eval, time: {:.4f}'.format(cost))
                logging.info('===== Validation ends =====\n\n')

            # Valid on some train data.
            if config.valid_train_data:
                logging.info('Valid first {} batches of train data.'
                             .format(config.max_valid_train_batch))
                _run_valid_perf(valid_train_reader,
                                train_batch_perf, train_eval_perf,
                                data_mode='train')
            # Valid on valid data.
            _run_valid_perf(valid_reader, batch_perf, eval_perf,
                            data_mode='valid')

            if config.test_when_valid:
                logging.info('===== Test starts =====')
                if config.valid_accord in {'all', 'batch'}:
                    start = time.time()
                    valid_model.clear_metric()
                    valid_model.run_epoch(sess, reader=test_reader,
                                          print_info=False,
                                          max_batch=config.max_test_batch)
                    cur_test = valid_model.print_metric()
                    batch_perf.update(cur_test, mode='test', log=True)
                    batch_perf.show()
                    cost = time.time() - start
                    logging.info('Test by batch, time: {:.4f}'.format(cost))
                if config.valid_accord in {'all', 'eval'}:
                    start = time.time()
                    cur_eval = valid_model.evaluate(sess, reader=test_reader,
                                                    print_info=False,
                                                    data_mode='test')
                    eval_perf.update(cur_eval, mode='test', log=True)
                    eval_perf.show()
                    cost = time.time() - start
                    logging.info('Test by eval, time: {:.4f}'.format(cost))
                logging.info('===== Test ends =====\n\n')

        logging.info('Validation once first.\n\n')
        valid_perf()

        # Start training.
        for epoch in range(config.max_epoch):
            start = time.time()
            logging.info('*' * 80)
            train_model.assign_lr(sess, lr)
            logging.info('===== Epoch {} starts, lr={} ====='.format(epoch, lr))
            train_model.clear_metric()

            avg_loss = train_model.run_epoch(sess, reader=train_reader,
                                             print_info=True,
                                             max_batch=config.max_train_batch)

            train_model.print_metric()

            lr = lr_adjust.adjust(epoch, lr, avg_loss)
            lr = max(lr, config.min_lr)
            cost = time.time() - start
            logging.info('Train time: {:.4f}'.format(cost))
            logging.info('===== Epoch {} ends =====\n\n'.format(epoch))

            if do_every(epoch, config.valid_every_n_epoch, config.max_epoch):
                valid_perf()
                if do_every(epoch, config.save_every_n_epoch, config.max_epoch):
                    if config.valid_accord == 'batch':
                        accord_perf = batch_perf
                    else:
                        accord_perf = eval_perf
                    if (accord_perf.best_valid_updated or
                            accord_perf.best_valid is None):
                        valid_model.on_valid_updated(sess)
                        self.save_model(global_step=epoch)

    def valid(self):
        """Valid model."""
        config = self.config
        sess = self.sess

        if not self.restored:
            self.restored = True
            self.valid_model = self.Model(config, mode='valid', reuse=False,
                                          **self.kwargs)

            restore_path = config.restore_path or config.model_path
            model_dir = os.path.dirname(restore_path)
            self.valid_model.restore_session(sess, model_dir)
        valid_model = self.valid_model

        logging.info('====== Validation starts ======')
        valid_model.clear_metric()
        avg_loss = valid_model.run_epoch(sess, print_info=False)
        logging.info('Validation:  average loss = {}'.format(avg_loss))
        valid_model.print_metric()
        logging.info('====== Validation ends ======')

    def infer(self):
        """Infer."""
        config = self.config
        sess = self.sess

        if not self.restored:
            self.restored = True
            self.infer_model = self.Model(config, mode='infer', reuse=False,
                                          **self.kwargs)
            restore_path = config.restore_path or config.model_path
            model_dir = os.path.dirname(restore_path)
            self.infer_model.restore_session(sess, model_dir)
        infer_model = self.infer_model

        infer_model.infer()

    def _prepare_lr_adjust(self):
        config = self.config

        lr_adjust = LRAdjust(config.lr, config.lr_decay,
                             config.lr_plateaus_thresh,
                             config.lr_plateaus_width)

        self._lr_adjust = lr_adjust

    def save_model(self, global_step=None):
        """Save model."""
        config = self.config
        model_path = config.model_path
        logging.info('Saving model to {} \n\n'.format(model_path))
        if not hasattr(self, 'saver'):
            saver = tf.train.Saver(tf.trainable_variables(),
                                   max_to_keep=config.max_to_keep)
            self.saver = saver
        self.saver.save(self.sess, model_path, global_step=global_step)
