"""Generate index file.

# CSharp

python3 -m api_emb.tool.codenn_index csharp
    --lower_api=T --api_idx_file=./data/api-lower.txt \
    --lower_doc=T --doc_idx_file=./data/word-lower.txt

python3 -m api_emb.tool.codenn_index csharp
    --lower_api=F --api_idx_file=./data/api.txt \
    --lower_doc=F --doc_idx_file=./data/word.txt

# SQL

python3 -m api_emb.tool.codenn_index sql \
    --lower_api=T --api_idx_file=./data/api-sql-lower.txt \
    --lower_doc=T --doc_idx_file=./data/word-sql-lower.txt

python3 -m api_emb.tool.codenn_index sql \
    --lower_api=F --api_idx_file=./data/api-sql.txt \
    --lower_doc=F --doc_idx_file=./data/word-sql.txt

# RNSQL: Rename table and column

python3 -m api_emb.tool.codenn_index rnsql \
    --lower_api=T --api_idx_file=./data/api-rnsql-lower.txt \
    --lower_doc=T --doc_idx_file=./data/word-rnsql-lower.txt

python3 -m api_emb.tool.codenn_index rnsql \
    --lower_api=F --api_idx_file=./data/api-rnsql.txt \
    --lower_doc=F --doc_idx_file=./data/word-rnsql.txt

"""
import codecs
import sys
from collections import Counter
from math import log10

from common.util import ConfigParser

from .split import split_csharp
from .split import split_sql
from .split import split_text


class CSharpConfig(ConfigParser):
    """CSharpConfig."""

    def __init__(self, args=[], prefix=''):
        super(CSharpConfig, self).__init__(prefix=prefix)
        self.lang = 'csharp'
        self.src_file = './data/codenn/data/stackoverflow/csharp/data/train.txt'

        self.api_idx_file = './data/1api.txt'
        self.lower_api = True
        self.doc_idx_file = './data/1word.txt'
        self.lower_doc = True

        self._parse_args(args)


class SQLConfig(ConfigParser):
    """SQLConfig."""

    def __init__(self, args=[], prefix=''):
        super(SQLConfig, self).__init__(prefix=prefix)
        self.lang = 'sql'
        self.src_file = './data/codenn/data/stackoverflow/sql/train.txt.tok'

        self.api_idx_file = './data/1api-sql.txt'
        self.lower_api = True
        self.doc_idx_file = './data/1word-sql.txt'
        self.lower_doc = True

        self._parse_args(args)


class RNSQLConfig(ConfigParser):
    """RNSQLConfig."""

    def __init__(self, args=[], prefix=''):
        super(RNSQLConfig, self).__init__(prefix=prefix)
        self.lang = 'rnsql'
        self.src_file = './data/codenn/data/stackoverflow/sql/train.txt.rn.tok'

        self.api_idx_file = './data/1api-rnsql.txt'
        self.lower_api = True
        self.doc_idx_file = './data/1word-rnsql.txt'
        self.lower_doc = True

        self._parse_args(args)


def text2words(text, lower=True):
    words = split_text(text, lower=lower)
    words = [w.strip() for w in words]
    return words


def code2tokens(code, lower=True, lang='csharp'):
    if lang == 'csharp':
        tokens = split_csharp(code)
    elif lang in {'sql', 'rnsql'}:
        assert lower is True
        tokens = split_sql(code)
    else:
        raise NotImplementedError()
    if lower:
        tokens = [t.lower().strip().replace('\n', ' \\n ') for t in tokens]
    else:
        tokens = [t.strip().replace('\n', ' \\n ') for t in tokens]
    return tokens


def main(config):
    apis = list()
    words = list()

    api_lines = {}
    word_lines = {}
    num_lines = 0.
    with codecs.open(config.src_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            num_lines += 1.
            _, _, doc, api, _ = line.split(sep='\t')
            cur_apis = code2tokens(api, lower=config.lower_api,
                                   lang=config.lang)
            cur_words = text2words(doc, lower=config.lower_doc)
            apis.extend(cur_apis)
            words.extend(cur_words)
            for a in set(cur_apis):
                api_lines[a] = api_lines.get(a, 0.) + 1.
            for w in set(cur_words):
                word_lines[w] = word_lines.get(w, 0.) + 1.

    apis_cnt = Counter(apis)
    words_cnt = Counter(words)

    apis_cnt = sorted(apis_cnt.items(),
                      key=lambda item: (item[1], item[0]), reverse=True)
    words_cnt = sorted(words_cnt.items(),
                       key=lambda item: (item[1], item[0]), reverse=True)

    for f, l, lines in zip([config.api_idx_file, config.doc_idx_file],
                         [apis_cnt, words_cnt],
                         [api_lines, word_lines]):
        with codecs.open(f, 'w', encoding='utf-8') as fout:
            for item in l:
                try:
                    t0, t1 = type(item[0]), type(item[1])
                    if t0 is not str:
                        print(t0, item)
                    idf = log10(num_lines / lines[item[0]])
                    fout.write('{}\t{}\t{:.6f}\n'.format(item[0], item[1], idf))
                except Exception as e:
                    print(e)


if __name__ == '__main__':
    configs = {
        'csharp': CSharpConfig,
        'sql': SQLConfig,
        'rnsql': RNSQLConfig,
    }
    config = configs[sys.argv[1]](args=sys.argv)
    main(config)
