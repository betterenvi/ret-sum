"""Model of learning API embedding."""
import logging
import tensorflow as tf

from common.util import walkdir
from common.model import Model
from model.nvdm import NVDM

from .index import Indexer
from .reader import Reader


distributions = tf.contrib.distributions


class ApiDocVAE(Model):
    """ApiDocVAE."""
    static_initialized = False

    @classmethod
    def static_initialize(cls, config):
        cls.api_indexer = Indexer(config.api_idx_file)
        cls.doc_indexer = Indexer(config.doc_idx_file)
        logging.info('Api index built with {} words.'.format(
            cls.api_indexer.num_chars))
        logging.info('Doc index built with {} words.'.format(
            cls.doc_indexer.num_chars))

    def _build_flow(self):
        config = self.config

        if not ApiDocVAE.static_initialized:
            ApiDocVAE.static_initialize(config)
            ApiDocVAE.static_initialized = True

        with tf.variable_scope('api'):
            self.api_vae = NVDM(config.api_config, self.is_train)

        with tf.variable_scope('doc'):
            self.doc_vae = NVDM(config.doc_config, self.is_train)

        self.ad_sim_loss = tf.reduce_sum(distributions.kl_divergence(
            self.api_vae.latent.distribution,
            self.doc_vae.latent.distribution))

        self.da_sim_loss = tf.reduce_sum(distributions.kl_divergence(
            self.doc_vae.latent.distribution,
            self.api_vae.latent.distribution))

        self.sim_loss = (self.ad_sim_loss + self.da_sim_loss) / 2

        self.loss = self.sim_loss
        self.loss += config.api_loss_wht * self.api_vae.loss
        self.loss += config.doc_loss_wht * self.doc_vae.loss
