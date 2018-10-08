import logging
import os
import tensorflow as tf

from tensorflow.contrib import distributions


class SoftSaver(tf.train.Saver):
    def restore(self, sess, save_path, soft=False,
                name_trans_func=None, prefixs=['']):
        ''' By default setting of tf.train.Saver, variables in models must
            exist in checkpoint file. If not, an error will occur.
            If soft == True, this constraint will be cancelled.

            Also support restore from the 'checkpoint' file directly.
            Args:
                * name_trans_func: a function to transform the names in the
                  checkpoint file to the names in the model
        '''
        if name_trans_func is None:
            name_trans_func = lambda x: x
        # support direct recovery from checkpoint file
        if os.path.basename(save_path) == 'checkpoint':
            with open(save_path) as f:
                first_line = next(f)  # model_checkpoint_path: "..."
                last_checkpoint_file_name = first_line.split()[-1].strip('"')
                save_path = os.path.join(os.path.dirname(save_path),
                                         last_checkpoint_file_name)

        if not soft:
            tf.train.Saver.restore(self, sess, save_path)
        else:
            ckpt = tf.contrib.framework.load_checkpoint(save_path)
            ckvar = [var_shape[0]
                     for var_shape
                     in tf.contrib.framework.list_variables(save_path)]
            ckvar = {name_trans_func(v): v for v in ckvar}
            glvar = [var.name.split(':')[0]
                     for var
                     in tf.trainable_variables()]

            prefix_glvar = []
            for var in glvar:
                for pre in prefixs:
                    if var.startswith(pre):
                        prefix_glvar.append(var)
                        continue

            intervar = set(ckvar.keys()) & set(prefix_glvar)

            scope = tf.get_variable_scope()
            with tf.variable_scope(scope, reuse=True):
                for var in intervar:
                    ckv = ckvar[var]
                    logging.info('Restoring {} as {}'.format(ckv, var))
                    sess.run(tf.assign(tf.get_variable(var),
                                       ckpt.get_tensor(ckv)))


def kl_divergence(distribution_a, distribution_b,
                  average_across_latent_dim=False,
                  average_across_batch=True):
    kl_div = distributions.kl_divergence(distribution_a, distribution_b)

    if average_across_latent_dim:
        kl_div = tf.reduce_mean(kl_div, axis=1)     # [b]
    else:
        kl_div = tf.reduce_sum(kl_div, axis=1)      # [b]

    if average_across_batch:
        kl_div = tf.reduce_mean(kl_div, axis=0)
    else:
        kl_div = tf.reduce_sum(kl_div, axis=0)

    return kl_div


def cosine_sim(a, b, average_across_batch=True):
    """Cosine similarity of tensor `a` and `b`.

    Args:
        a: [b, h]
        b: [b, h]
    Returns:
        s: scalar
    """
    norm_a = tf.norm(a, ord=2, axis=1)
    norm_b = tf.norm(b, ord=2, axis=1)

    sim = tf.reduce_sum(a * b, axis=1)
    sim = sim / (norm_a + 1e-12) / (norm_b + 1e-12)

    if average_across_batch:
        sim = tf.reduce_mean(sim, axis=0)
    else:
        sim = tf.reduce_sum(sim, axis=0)

    return sim


def word_dropout(idxs, seq_lens, unk_idx, keep_prob):
    """Map some words to <unk>.

    idxs: [b, t]
    seq_lens: [b]
    unk_idx: scalar
    """
    shape = tf.shape(idxs)
    b, t = shape[0], shape[1]
    unk = tf.constant([[unk_idx]], dtype=tf.int32)
    unk = tf.tile(unk, [b, t])

    in_range = tf.sequence_mask(seq_lens, maxlen=t, dtype=tf.bool)
    rand = tf.random_uniform(shape)
    drop = tf.greater(rand, keep_prob)
    drop = tf.logical_and(drop, in_range)

    ret = tf.where(drop, unk, idxs)
    return ret
