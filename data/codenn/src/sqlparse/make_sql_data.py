"""Tokenize SQL code.

Run at the path of this file:
cd data/codenn/src/sqlparse/

# if rename table and column:
python2 make_sql_data.py rn

# else:
python2 make_sql_data.py

"""
# -*- coding=utf-8 -*-
import codecs
import os
import sys
from SqlTemplate import SqlTemplate
reload(sys)
sys.setdefaultencoding('utf-8')

d = '../../data/stackoverflow/sql/'
to_make = ['train.txt', 'valid.txt', 'test.txt', 'dev/dev.txt', 'eval/eval.txt']
# to_make = ['valid.txt']
fs = []

for f in to_make:
    fs.append(os.path.join(d, f))


def make(f, rename_tab_col=False):
    if rename_tab_col:
        dst = f + '.rn.tok'
    else:
        dst = f + '.tok'
    print('Processing {} -> {}'.format(f, dst))
    with codecs.open(dst, mode='w', encoding='utf-8') as fout:
        with codecs.open(f, mode='r', encoding='utf-8') as fin:
            for i, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue
                fields = line.split('\t')
                # print line
                # print fields
                assert len(fields) == 5

                try:
                    q = SqlTemplate(fields[3], regex=True, rename=True,
                                    rename_tab_col=rename_tab_col)
                except Exception, e:
                    print i, line
                    print e
                sql = ' '.join(q.parseSql())
                fields[3] = sql

                out = '\t'.join(fields)

                fout.write(out + '\n')


if __name__ == '__main__':
    rename_tab_col = False if len(sys.argv) <= 1 else True
    for f in fs:
        make(f, rename_tab_col)
