import tensorflow as tf

from common.op import word_dropout


def make_decoder_hinputs(inputs, seq_lens, embedding, unk_id,
                         input_keep_prob, start_tokens):
    """Prepare hidden inputs for RNN based decoder.

    inputs: [b, t]
    seq_lens: [b]
    embedding: [v, d]
    unk_id: scalar
    start_tokens: [b]
    """
    rem_idx_inputs = word_dropout(inputs, seq_lens, unk_id, input_keep_prob)
    hinputs = tf.nn.embedding_lookup(embedding, rem_idx_inputs)
    b_shape = tf.shape(hinputs)[0]
    t_shape = tf.shape(hinputs)[1]
    h_shape = hinputs.shape[2]
    start_tokens = tf.reshape(start_tokens, [b_shape, 1])
    start_hinputs = tf.nn.embedding_lookup(embedding, start_tokens)
    hinputs = tf.concat([start_hinputs, hinputs], axis=1)
    hinputs = tf.slice(hinputs, [0, 0, 0], [-1, t_shape, -1])
    hinputs = tf.reshape(hinputs, [b_shape, t_shape, h_shape])
    return hinputs


def get_rnn_cell(cell_type, hidden_size, context=None):
    if cell_type == 'gru':
        cell = tf.contrib.rnn.GRUCell(hidden_size)
    elif cell_type == 'cgru':
        assert context is not None
        cell = CtxGRUCell(hidden_size, context)
    elif cell_type == 'lstm':
        cell = tf.contrib.rnn.LSTMCell(hidden_size)
    else:
        raise NotImplementedError()
    return cell
