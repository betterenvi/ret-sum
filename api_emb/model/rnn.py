"""RNN based encoder."""
import tensorflow as tf

from .base import BaseSeqModel


class RNNModel(BaseSeqModel):
    """RNNModel."""

    def _build_encoder(self):
        """Inference Network, i.e. q(h|X)."""
        config = self.config

        hinputs = self.hinputs

        def single_cell():
            cell = tf.contrib.rnn.GRUCell(config.encoder_hidden_size)

            if self.is_train and config.keep_prob < 1.:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)

            return cell

        cell_fw = tf.contrib.rnn.MultiRNNCell(
            [single_cell() for _ in range(config.num_rnn_encoder_layers)],
            state_is_tuple=False)

        cell_bw = tf.contrib.rnn.MultiRNNCell(
            [single_cell() for _ in range(config.num_rnn_encoder_layers)],
            state_is_tuple=False)

        _, output_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            hinputs,
            sequence_length=self.seq_lens,
            initial_state_fw=None,
            initial_state_bw=None,
            dtype=tf.float32,
            time_major=False)

        output_state_fw, output_state_bw = output_states

        output = tf.concat([output_state_fw, output_state_bw], axis=1)

        self.encoder_logits = output

    def _build_decoder(self):
        super()._build_decoder()
