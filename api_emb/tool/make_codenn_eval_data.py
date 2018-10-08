"""Make data for codenn evaluation.


# CSharp
python3 -m api_emb.tool.make_codenn_eval_data csharp

# SQL
python3 -m api_emb.tool.make_codenn_eval_data sql

"""
import codecs
import collections
import os
import random
import sys

from common.util import ConfigParser


class CSharpConfig(ConfigParser):
    def __init__(self, args=[]):
        super(CSharpConfig, self).__init__()
        d = './data/codenn/data/stackoverflow/csharp/'
        self.dev_src_ref_path = os.path.join(d, 'dev/ref.txt')
        self.dev_dst_ref_path = os.path.join(d, 'dev/ref-ann.txt')
        self.eval_src_ref_path = os.path.join(d, 'eval/ref.txt')
        self.eval_dst_ref_path = os.path.join(d, 'eval/ref-ann.txt')
        self.shuffle = True

        self.full_data_path = os.path.join(d, 'test.txt')


class SQLConfig(ConfigParser):
    def __init__(self, args=[]):
        super(SQLConfig, self).__init__()
        d = './data/codenn/data/stackoverflow/sql/'
        self.dev_src_ref_path = os.path.join(d, 'dev/ref.txt')
        self.dev_dst_ref_path = os.path.join(d, 'dev/ref-ann.txt')
        self.eval_src_ref_path = os.path.join(d, 'eval/ref.txt')
        self.eval_dst_ref_path = os.path.join(d, 'eval/ref-ann.txt')
        self.shuffle = True

        self.full_data_path = os.path.join(d, 'test.txt')


def main(config):
    full_docs = collections.defaultdict(list)
    with open(config.full_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, cid, doc, api, _ = line.split(sep='\t')

            doc = doc.lower()
            full_docs[cid].append(doc)

    def _make(src, dst):
        with codecs.open(dst, 'w', encoding='utf-8') as fout:
            with codecs.open(src, 'r', encoding='utf-8') as fin:
                lines = fin.readlines()
                if config.shuffle:
                    print('Shuffling...')
                    random.shuffle(lines)
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    cid, doc = line.split('\t')

                    if doc.lower() not in full_docs[cid]:
                        fout.write('{}\t{}\n'.format(cid, doc))

    _make(config.dev_src_ref_path, config.dev_dst_ref_path)
    _make(config.eval_src_ref_path, config.eval_dst_ref_path)


if __name__ == '__main__':
    configs = {
        'csharp': CSharpConfig,
        'sql': SQLConfig,
    }
    config = configs[sys.argv[1]](args=sys.argv)
    main(config)
