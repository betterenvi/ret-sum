"""Define configs for models."""
import numpy as np

from common.util import ConfigParser


class ModelConfig(ConfigParser):
    """ModelConfig."""

    def __init__(self, args=[], prefix='', check=True):
        """Init ModelConfig."""
        super(ModelConfig, self).__init__(prefix=prefix, check=False)

        # Model.
        self.any_batch_size = True
        self.batch_size = 64

        # Data.
        ## Train data.
        self.train_dir = '^-^'
        self.train_ext = ''
        self.train_exact = False
        self.train_prefetch = False
        ## Valid data.
        self.valid_dir = '^-^'
        self.valid_ext = ''
        self.valid_exact = False
        self.valid_prefetch = False
        ## Test data.
        self.test_dir = '^-^'
        self.test_ext = ''
        self.test_exact = False
        self.test_prefetch = False

        # Training.
        self.optimizer = 'adam'
        self.lr = 0.005
        self.lr_decay = 0.8
        self.lr_decay_begin = 5
        self.lr_plateaus_thresh = 0.05
        self.lr_plateaus_width = 3
        self.min_lr = 1e-10
        self.max_epoch = 10000
        self.clip_grad = 5.
        self.print_every_n_batch = 5
        # Restore.
        self.continue_run = True
        self.restore_path = ''              # if '', will set to model_path.

        self.valid_train_data = True        # do validation on train data.
        self.max_valid_train_batch = 32     # num of batched to valid train.

        # Validation.
        self.valid_accord = 'eval'      # batch, eval, or all
        self.valid_every_n_epoch = 1
        self.test_when_valid = False

        self.max_train_batch = np.inf
        self.max_valid_batch = np.inf
        self.max_test_batch = np.inf

        # Save.
        self.model_path = './model/'
        self.save_every_n_epoch = 1
        self.max_to_keep = 1
        self.config_file = 'config.json'    # Will be put under model_path.

        # Controller.
        self.mode = 'train'

        self.log_level = 'info'
        self.log_to_file = False
        self.log_file = 't.log'

        self._parse_args(args, check=check)


    def _check(self):
        super()._check()
        assert self.valid_accord in {'all', 'batch', 'eval'}
