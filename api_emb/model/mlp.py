"""MLP based encoder and decoder."""
import tensorflow as tf

from tensorflow.contrib.layers import fully_connected as fc_layer

from .base import BaseSeqModel


class MLPModel(BaseSeqModel):
    """MLPModel."""

    def _build_encoder(self):
        """Inference Network, i.e. q(h|X)."""
        config = self.config

        self.encoder_hiddens = []

        with tf.variable_scope('encoder'):
            if config.average_embedding_by_idf:
                hinputs = self._mask(self.hinputs, by='weight')
            else:
                hinputs = self._mask(self.hinputs, by='seq_len')

            if config.max_embedding:
                hinputs = tf.reduce_max(hinputs, axis=1)
            else:
                hinputs = tf.reduce_sum(hinputs, axis=1)
                if config.average_embedding:
                    if config.average_embedding_by_idf:
                        n = tf.reduce_sum(self.weights, axis=1, keep_dims=True)
                    else:
                        n = tf.expand_dims(tf.cast(self.seq_lens, tf.float32),
                                           axis=1)
                    hinputs = tf.nn.tanh(hinputs / n)

            for lid in range(config.num_encoder_layers):
                outputs = fc_layer(hinputs,
                                   config.encoder_hidden_size,
                                   activation_fn=tf.nn.tanh)
                if config.use_residual:
                    hinputs += outputs
                else:
                    hinputs = outputs
                self.encoder_hiddens.append(hinputs)

            self.encoder_logits = hinputs
