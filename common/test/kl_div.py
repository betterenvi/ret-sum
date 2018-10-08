import numpy as np
import tensorflow as tf

from tensorflow.contrib import distributions

from ..util import pair_kl_divergence


class TestKL(tf.test.TestCase):
    """TestKL"""

    def testKL(self):
        mu1, sd1, mu2, sd2 = [np.random.rand(4, 6) for _ in range(4)]
        pair_kl = pair_kl_divergence(mu1, sd1, mu2, sd2)

        dist1 = distributions.Normal(mu1, sd1)
        dist2 = distributions.Normal(mu2, sd2)
        kl_tf = distributions.kl_divergence(dist1, dist2)

        with tf.Session() as sess:
            kl_val = sess.run(kl_tf)
            kl_val = kl_val.sum(axis=-1)

        self.assertAllClose(np.diag(pair_kl), kl_val)


if __name__ == '__main__':
    tf.test.main()
