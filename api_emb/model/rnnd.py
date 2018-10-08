"""RNN based encoder and decoder."""
import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import fully_connected as fc_layer
from tensorflow.python.layers.core import Dense

from common.op import word_dropout

from .rnn import RNNModel

from ..metric.bleu import get_bleu


class RNNDModel(RNNModel):
    """RNNDModel."""

    def _build_decoder(self):
        config = self.config
        ad_latent = self.kwargs['ad_latent']

        if ad_latent is None:
            if self.is_train:
                sampled = self.latent
                maximum_iterations = tf.shape(self.idx_inputs)[1]
            else:
                sampled = self.latent_mu
                maximum_iterations = tf.shape(self.idx_inputs)[1]
                if self.mode == 'test':
                    maximum_iterations = config.max_gen_seq_len
        else:
            sampled = ad_latent
            maximum_iterations = \
                tf.shape(self.idx_inputs)[1] if self.is_train \
                else config.max_gen_seq_len

        b_shape = tf.shape(sampled)[0]

        initial_state = fc_layer(sampled, config.decoder_hidden_size,
                                 activation_fn=tf.nn.tanh)

        indexer = self.kwargs['indexer']
        word2id = indexer.word2id
        sos_id = word2id(indexer.SOS)
        eos_id = word2id(indexer.EOS)
        unk_id = word2id(indexer.UNK)

        start_tokens = tf.constant([sos_id], dtype=tf.int32)
        start_tokens = tf.tile(start_tokens, [b_shape])     # [b]

        def single_cell(context):
            self.logger.info('Use {} cell as rnnd cell.'
                             .format(config.decoder_cell_type))
            if config.decoder_cell_type == 'gru':
                cell = tf.contrib.rnn.GRUCell(config.decoder_hidden_size)
            else:
                raise NotImplementedError()
            return cell

        output_layer = Dense(config.vocab_size)

        if self.is_train:
            # Apply input dropout.
            self.rem_idx_inputs = word_dropout(
                self.idx_inputs,
                self.seq_lens,
                unk_id,
                config.input_keep_prob)
            hinputs = tf.nn.embedding_lookup(self.embedding,
                                             self.rem_idx_inputs)
            t_shape = tf.shape(hinputs)[1]
            h_shape = hinputs.shape[2]
            start_tokens = tf.reshape(start_tokens, [b_shape, 1])
            start_hinputs = tf.nn.embedding_lookup(self.embedding, start_tokens)
            inputs = tf.concat([start_hinputs, hinputs], axis=1)
            inputs = tf.slice(inputs, [0, 0, 0], [-1, t_shape, -1])
            inputs = tf.reshape(inputs, [b_shape, t_shape, h_shape])
            self.helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=inputs,
                sequence_length=self.seq_lens,
                time_major=False)
            cell = single_cell(initial_state)

            if config.num_rnn_decoder_layers > 1:
                initial_state = [initial_state]
                cell = [cell]
                for l in range(1, config.num_rnn_decoder_layers):
                    initial_state_l = fc_layer(sampled,
                                               config.decoder_hidden_size,
                                               activation_fn=tf.nn.tanh)
                    initial_state.append(initial_state_l)
                    cell_l = single_cell(initial_state_l)
                    cell.append(cell_l)
                cell = tuple(cell)
                initial_state = tuple(initial_state)
                cell = tf.contrib.rnn.MultiRNNCell(cell, state_is_tuple=True)

            self.seq_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell,
                self.helper,
                initial_state,
                output_layer)
            self.final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                self.seq_decoder,
                output_time_major=False,
                impute_finished=False,
                maximum_iterations=maximum_iterations)
            # [batch_size, max_time_step, num_decoder_symbols]
            self.seq_logits = self.final_outputs.rnn_output
        else:
            initial_state = tf.contrib.seq2seq.tile_batch(
                initial_state, multiplier=config.beam_width)
            cell = single_cell(initial_state)

            if config.num_rnn_decoder_layers > 1:
                initial_state = [initial_state]
                cell = [cell]
                for l in range(1, config.num_rnn_decoder_layers):
                    initial_state_l = fc_layer(sampled,
                                               config.decoder_hidden_size,
                                               activation_fn=tf.nn.tanh)
                    initial_state_l = tf.contrib.seq2seq.tile_batch(
                        initial_state_l, multiplier=config.beam_width)
                    initial_state.append(initial_state_l)
                    cell_l = single_cell(initial_state_l)
                    cell.append(cell_l)
                cell = tuple(cell)
                initial_state = tuple(initial_state)
                cell = tf.contrib.rnn.MultiRNNCell(cell, state_is_tuple=True)

            self.seq_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell,
                self.embedding,
                start_tokens,
                eos_id,
                initial_state,
                config.beam_width,
                output_layer=output_layer,
                length_penalty_weight=config.length_penalty_weight)

            self.predict_decoder_outputs, _, self.final_sequence_lengths = \
                tf.contrib.seq2seq.dynamic_decode(
                    self.seq_decoder,
                    output_time_major=False,
                    impute_finished=False,
                    maximum_iterations=maximum_iterations + 20)
            # [batch_size, max_time_step, beam_width]
            self.predicts = self.predict_decoder_outputs.predicted_ids

    def _build_log_prob(self):
        self.targets = self.idx_inputs
        self.log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.targets,
            logits=self.seq_logits)     # [b, t], not masked yet.

    def _build_loss(self):
        if self.is_train:
            super()._build_loss()
        else:
            self._build_kl_div()
            self.loss = self.kl_div

    def _build_targets_and_metrics(self):
        if self.is_train:
            super()._build_targets_and_metrics()
        else:
            self.perf_key = 'kl_div'
            self.perf_large_is_better = False
            self.metric_keys = ['kl_div']
            for k, v in zip(self.metric_keys, [self.kl_div]):
                self.run_targets[k] = v

            self.run_targets['latent_mu'] = self.latent_mu
            self.run_targets['latent_sd'] = self.latent_sd

    def ids2texts(self, predicts, lens):
        """
        predicts: [b, t]
        lens: [b]
        """
        config = self.config
        pred_texts = []
        nounk_pred_texts = []
        indexer = self.kwargs['indexer']
        word2id = indexer.word2id
        id2word = indexer.id2word
        eos_id = word2id(indexer.EOS)
        unk_id = word2id(indexer.UNK)
        for ids, ln in zip(predicts, lens):
            ids = list(ids)
            # Cut redundant ids. Cut <eos>, too.
            try:
                l = min(ids.index(eos_id), ln - 1)
            except:
                l = ln - 1
            l = max(0, l)
            words = [id2word(idx) for idx in ids[:l]]
            pred_texts.append(' '.join(words))
            nounk_words = [id2word(idx) for idx in ids[:l] if idx != unk_id]
            nounk_pred_texts.append(' '.join(nounk_words))
        return pred_texts, nounk_pred_texts

    def evaluate_bleu(self, sess, reader=None,
                      print_info=False, batch_data_offset=0,
                      data_mode='valid'):
        config = self.config

        reader = reader or self.reader
        offset = batch_data_offset

        predicts = []
        lens = []

        gold_texts = []
        for batch, batch_data in enumerate(reader):
            fd = {
                self.idx_inputs: batch_data[offset + 0],
                self.weights: batch_data[offset + 1],
                self.seq_lens: batch_data[offset + 2],
                self.bow_inputs: batch_data[offset + 4],
            }
            cur_predicts, cur_lens, latent_mu, latent_sd = sess.run(
                [self.predicts, self.final_sequence_lengths,
                 self.latent_mu, self.latent_sd],
                feed_dict=fd)

            # Only save the best one.
            predicts.extend(cur_predicts[:, :, 0])
            lens.extend(cur_lens[:, 0])

            gold_texts.extend(batch_data[offset + 3])

        _, pred_texts = self.ids2texts(predicts, lens)
        gold_texts = [[' '.join(text)] for text in gold_texts]

        bleu = get_bleu(pred_texts, gold_texts)
        self.logger.info('BLEU: ' + ' '.join([str(v) for v in bleu]))
        return bleu[0]

    def evaluate(self, sess, reader=None,
                 print_info=False, batch_data_offset=0, data_mode='valid'):

        return self.evaluate_bleu(sess, reader=reader,
                                  print_info=print_info,
                                  batch_data_offset=batch_data_offset,
                                  data_mode=data_mode)
