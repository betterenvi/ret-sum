import codecs
import sys
from common.util import ConfigParser


class Config(ConfigParser):
    """Config."""

    def __init__(self, args=[], prefix='', sep='/'):
        super(Config, self).__init__(prefix=prefix, sep=sep)
        self.src_file = 'C:/Users/CQY/Desktop/tmp.txt'

        self.api_idx_file = './data/api.txt'
        self.doc_idx_file = './data/word.txt'

        self._parse_args(args)


def main(config):
    apis = set()
    words = set()

    with codecs.open(config.src_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            api_seq, _, doc = line.split(sep=':', maxsplit=2)
            apis.update(api_seq.split())
            words.update(doc.split())

    apis = sorted(apis)
    words = sorted(words)

    for f, l in zip([config.api_idx_file, config.doc_idx_file], [apis, words]):
        with codecs.open(f, 'w', encoding='utf-8') as fout:
            for a in l:
                fout.write(a + '\n')


if __name__ == '__main__':
    config = Config(args=sys.argv)
    main(config)
