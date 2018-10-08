import os
from common.util import ConfigParser


class SQLDataBase(ConfigParser):
    def __init__(self, args=[], prefix='', check=True):
        super(SQLDataBase, self).__init__(prefix=prefix, check=False)
        # Index files.
        self.api_idx_file = './data/api-sql.txt'
        self.doc_idx_file = './data/word-sql.txt'
        self.api_idx_file_lower = './data/api-sql-lower.txt'
        self.doc_idx_file_lower = './data/word-sql-lower.txt'
        self.lower_api_idx = True   # if true, api_idx_file->api_idx_file_lower
        self.lower_doc_idx = True
        self.min_api_cnt = 3
        self.min_doc_cnt = 3

        d = './data/codenn/data/stackoverflow/sql/'
        self.codenn_dev_ref = os.path.join(d, 'dev/ref-ann.txt')
        self.codenn_eval_ref = os.path.join(d, 'eval/ref-ann.txt')
        # Summary refs.
        self.codenn_sum_dev_ref = os.path.join(d, 'dev/ref.txt')
        self.codenn_sum_eval_ref = os.path.join(d, 'eval/ref.txt')
        self.codenn_sum_dev_api = os.path.join(d, 'dev/dev.txt.tok')
        self.codenn_sum_eval_api = os.path.join(d, 'eval/eval.txt.tok')

        self._parse_args(args, check=check)

    def _check(self):
        super()._check()
        if self.lower_api_idx:
            self.api_idx_file = self.api_idx_file_lower
        if self.lower_doc_idx:
            self.doc_idx_file = self.doc_idx_file_lower


class SQLData(SQLDataBase):
    def __init__(self, args=[], prefix='', check=True):
        super(SQLData, self).__init__(prefix=prefix, check=False)
        # Train data.
        # '.tok' denotes 'pre-tokenized' code.
        self.train_dir = './data/codenn/data/stackoverflow/sql/'
        self.train_ext = 'train.txt.tok'
        self.train_exact = False
        self.train_prefetch = True
        # Valid data.
        self.valid_dir = './data/codenn/data/stackoverflow/sql/'
        self.valid_ext = 'valid.txt.tok'
        self.valid_exact = False
        self.valid_prefetch = True
        # Test data.
        self.test_dir = './data/codenn/data/stackoverflow/sql/'
        self.test_ext = 'test.txt.tok'
        self.test_exact = False
        self.test_prefetch = True
        self._parse_args(args, check=check)


class RNSQLDataBase(ConfigParser):
    def __init__(self, args=[], prefix='', check=True):
        super(RNSQLDataBase, self).__init__(prefix=prefix, check=False)
        # Index files.
        self.api_idx_file = './data/api-rnsql.txt'
        self.doc_idx_file = './data/word-rnsql.txt'
        self.api_idx_file_lower = './data/api-rnsql-lower.txt'
        self.doc_idx_file_lower = './data/word-rnsql-lower.txt'
        self.lower_api_idx = True   # if true, api_idx_file->api_idx_file_lower
        self.lower_doc_idx = True
        self.min_api_cnt = 3
        self.min_doc_cnt = 3

        d = './data/codenn/data/stackoverflow/sql/'
        self.codenn_dev_ref = os.path.join(d, 'dev/ref-ann.txt')
        self.codenn_eval_ref = os.path.join(d, 'eval/ref-ann.txt')
        # Summary refs.
        self.codenn_sum_dev_ref = os.path.join(d, 'dev/ref.txt')
        self.codenn_sum_eval_ref = os.path.join(d, 'eval/ref.txt')
        self.codenn_sum_dev_api = os.path.join(d, 'dev/dev.txt.rn.tok')
        self.codenn_sum_eval_api = os.path.join(d, 'eval/eval.txt.rn.tok')

        self._parse_args(args, check=check)

    def _check(self):
        super()._check()
        if self.lower_api_idx:
            self.api_idx_file = self.api_idx_file_lower
        if self.lower_doc_idx:
            self.doc_idx_file = self.doc_idx_file_lower


class RNSQLData(RNSQLDataBase):
    def __init__(self, args=[], prefix='', check=True):
        super(RNSQLData, self).__init__(prefix=prefix, check=False)
        # Train data.
        # '.tok' denotes 'pre-tokenized' code.
        self.train_dir = './data/codenn/data/stackoverflow/sql/'
        self.train_ext = 'train.txt.rn.tok'
        self.train_exact = False
        self.train_prefetch = True
        # Valid data.
        self.valid_dir = './data/codenn/data/stackoverflow/sql/'
        self.valid_ext = 'valid.txt.rn.tok'
        self.valid_exact = False
        self.valid_prefetch = True
        # Test data.
        self.test_dir = './data/codenn/data/stackoverflow/sql/'
        self.test_ext = 'test.txt.rn.tok'
        self.test_exact = False
        self.test_prefetch = True
        self._parse_args(args, check=check)
