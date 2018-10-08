"""Config."""
import itertools
import os

from common.config import ModelConfig


class ApiDocVAEConfig(ModelConfig):
    """ApiDocVAEConfig."""

    def __init__(self, args=[], prefix='', check=True):
        super(ApiDocVAEConfig, self).__init__(prefix=prefix, check=False)

        # Task.
        self.task = 'ret'   # 'ret' for code retrieval, 'sum' for code summary.
        self.lang = 'csharp'

        # Model option.
        self.api_model = 'mlp'
        self.doc_model = 'mlp'

        self.any_batch_size = True
        self.batch_size = 64

        # Files used for building index.
        self.api_idx_file = './data/api.txt'
        self.doc_idx_file = './data/word.txt'
        self.api_idx_file_lower = './data/api-lower.txt'
        self.doc_idx_file_lower = './data/word-lower.txt'
        self.lower_api_idx = True   # if true, api_idx_file->api_idx_file_lower
        self.lower_doc_idx = True
        self.min_api_cnt = 3
        self.min_doc_cnt = 3

        # Data.
        # Train data.
        self.train_dir = './data/codenn/data/stackoverflow/csharp/data/'
        self.train_ext = 'train.txt'
        self.train_exact = False
        self.train_prefetch = True
        # Valid data.
        self.valid_dir = './data/codenn/data/stackoverflow/csharp/data/'
        self.valid_ext = 'valid.txt'
        self.valid_exact = False
        self.valid_prefetch = True
        # Test data.
        self.test_dir = './data/codenn/data/stackoverflow/csharp/data/'
        self.test_ext = 'test.txt'
        self.test_exact = False
        self.test_prefetch = True

        # Data format.

        self.switch_data_to = ''

        # Codenn dev and eval ref data.
        # Retrieval refs.
        d = './data/codenn/data/stackoverflow/csharp/'
        self.codenn_dev_ref = os.path.join(d, 'dev/ref-ann.txt')
        self.codenn_eval_ref = os.path.join(d, 'eval/ref-ann.txt')
        # Summary refs.
        self.codenn_sum_dev_ref = os.path.join(d, 'dev/ref.txt')
        self.codenn_sum_eval_ref = os.path.join(d, 'eval/ref.txt')
        self.codenn_sum_dev_api = os.path.join(d, 'dev/dev.txt')
        self.codenn_sum_eval_api = os.path.join(d, 'eval/eval.txt')

        self.meteor_jar_path = '~/meteor-1.5/meteor-1.5.jar'

        # Pickle.
        self.pickle_data = True
        self.re_pickle = True  # If True, will re-generate pickle files.

        # Loss.
        self.use_variational = True
        self.api_loss_wht = 0.25
        self.doc_loss_wht = 0.25
        self.sim_loss_option = 'klm'     # kl-mean . or 'kl'
        self.use_fix_sd_in_klm = False
        self.klm_latent_prior_sd = 1.0
        # Divergence annealing.

        # Eval.
        # Retrieval task.
        self.num_eval_times = 20
        self.num_candidates = 50
        self.confidence = 0.95

        # Control.
        self.mode = 'train'
        self.optimizer = 'Adam'
        self.lr = 0.005
        self.print_every_n_batch = 200

        # Save.

        # Restore.
        self.restore_api = True
        self.restore_doc = True

        self._parse_args(args, check=check)

    def _check(self):
        super()._check()
        assert self.task in {'ret', 'sum'}

        assert 0. <= (self.api_loss_wht + self.doc_loss_wht) <= 1.

        if self.lower_api_idx:
            self.api_idx_file = self.api_idx_file_lower
        if self.lower_doc_idx:
            self.doc_idx_file = self.doc_idx_file_lower

        if 'sql' in self.switch_data_to:
            assert self.lang == 'sql'

    def switch_data(self, data_config):
        # Switch data if needed.
        for m, k in itertools.product(['train', 'valid', 'test'],
                                      ['dir', 'ext', 'exact', 'prefetch']):
            name = '_'.join([m, k])
            self.__dict__[name] = data_config.__dict__[name]

        for attr in ['codenn_dev_ref', 'codenn_eval_ref',
                     'codenn_sum_dev_ref', 'codenn_sum_eval_ref',
                     'codenn_sum_dev_api', 'codenn_sum_eval_api',
                     # Index files:
                     'api_idx_file', 'doc_idx_file',
                     'api_idx_file_lower', 'doc_idx_file_lower',
                     'lower_api_idx', 'lower_doc_idx',
                     'min_api_cnt', 'min_doc_cnt']:
            if hasattr(data_config, attr):
                self.__dict__[attr] = data_config.__dict__[attr]


class BaseConfig(ModelConfig):
    """BaseConfig for all models."""

    def __init__(self, args=[], prefix='', check=True):
        super(BaseConfig, self).__init__(prefix=prefix, check=False)

        self.any_num_steps = True
        self.num_steps = 64
        self.vocab_size = None

        # Embedding.
        self.embedding_size = 256

        # Encoder.
        self.num_encoder_layers = 2
        self.train_encoder = True

        # Latent.
        self.latent_size = 128
        self.latent_prior_sd = 1.
        self.use_posterior_mu = False

        # Decoder.
        self.num_decoder_layers = 1
        self.decoder_hidden_size = 256
        # self.train_decoder = True

        # Loss.
        self.use_variational = True
        self.average_across_timesteps = False
        self.kl_div_wht = 0.0
        self.use_idf = False    # use idf as weights.
        self.eos_wht = ''

        self._parse_args(args, check=check)

    def _check(self):
        super()._check()


# Configs for MLP models.
class MLPConfig(BaseConfig):
    """MLPConfig."""

    def __init__(self, args=[], prefix='', check=True):
        """Init MLPConfig."""
        super(MLPConfig, self).__init__(prefix=prefix, check=False)

        # Encoder.
        self.embedding_size = 256
        self.average_embedding = True
        self.average_embedding_by_idf = False
        self.max_embedding = False
        self.num_encoder_layers = 2
        self.encoder_hidden_size = 256
        self.latent_size = 128
        self.use_residual = False

        # Decoder.
        self.num_decoder_layers = 1
        self.decoder_hidden_size = 256

        # Loss.
        self.eos_wht = ''

        self._parse_args(args, check=check)

    def _check(self):
        super()._check()
        assert 0. <= self.kl_div_wht <= 1.
        if self.average_embedding_by_idf:
            assert self.use_idf
        if self.use_residual:
            assert self.encoder_hidden_size == self.embedding_size


class ApiMLPConfig(MLPConfig):
    """ApiMLPConfig."""

    def __init__(self, args=[], prefix='api', check=True):
        super(ApiMLPConfig, self).__init__(prefix=prefix, check=False)
        self.num_steps = 1000
        self._parse_args(args, check=check)

    def _check(self):
        super()._check()


class DocMLPConfig(MLPConfig):
    """DocMLPConfig."""

    def __init__(self, args=[], prefix='doc', check=True):
        super(DocMLPConfig, self).__init__(prefix=prefix, check=False)
        self.num_steps = 64
        self._parse_args(args, check=check)


class ApiMLPERNNDConfig(MLPConfig):
    """ApiMLPERNNDConfig."""

    def __init__(self, args=[], prefix='doc', check=True):
        super(ApiMLPERNNDConfig, self).__init__(prefix=prefix, check=False)

        # Decoder.
        self.decoder_hidden_size = 256
        self.num_rnn_decoder_layers = 1
        self.input_keep_prob = 0.2
        self.decoder_cell_type = 'gru'

        # For test.
        self.beam_width = 10
        self.length_penalty_weight = 0.
        self.max_gen_seq_len = 200
        self.remove_unk_in_pred = True

        self._parse_args(args, check=check)


class DocMLPERNNDConfig(MLPConfig):
    """DocMLPERNNDConfig."""

    def __init__(self, args=[], prefix='doc', check=True):
        super(DocMLPERNNDConfig, self).__init__(prefix=prefix, check=False)

        # Decoder.
        self.decoder_hidden_size = 256
        self.num_rnn_decoder_layers = 1
        self.input_keep_prob = 0.2
        self.decoder_cell_type = 'gru'

        # For test.
        self.beam_width = 10
        self.length_penalty_weight = 0.
        self.max_gen_seq_len = 20
        self.remove_unk_in_pred = True

        self._parse_args(args, check=check)
