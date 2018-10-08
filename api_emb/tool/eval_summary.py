r"""Evaluate METEOR score.

DIR=./data/codenn/data/stackoverflow/csharp/
python3 -m api_emb.tool.eval_summary \
    --gold=${DIR}/eval/ref.txt \
    --pred=${DIR}/eval/codenn.txt

"""
import argparse
import codecs
import collections

from ..metric.bleu import get_bleu
from ..metric.meteor import get_meteor


def str2bool(s):
    return s.lower() not in ['false', 'f', '0', 'none', 'no', 'n']


parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gold', type=str)
parser.add_argument('-p', '--pred', type=str)
parser.add_argument('-r', '--numref', type=int, default=3)
parser.add_argument('-j', '--jar_path', type=str,
                    default='~/meteor-1.5/meteor-1.5.jar')
parser.add_argument('-n', '--norm', type=str2bool, default=False)
parser.add_argument('-t', '--pre_tokenize', type=str2bool, default=True)
config = parser.parse_args()


def read_ref(ref_path):
    ref = collections.defaultdict(list)
    with codecs.open(ref_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rid, txt = line.split('\t', maxsplit=1)
            ref[rid].append(txt)
    return ref


def read_pred(pred_path):
    pred = {}
    with codecs.open(pred_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rid, txt = line.split('\t', maxsplit=1)
            pred[rid] = txt
    return pred


def main(config):
    ref = read_ref(config.gold)
    pred = read_pred(config.pred)
    assert set(ref.keys()) == set(pred.keys())

    keys = ref.keys()
    ref_texts = [ref[k] for k in keys]
    pred_texts = [pred[k] for k in keys]

    meteor = get_meteor(pred_texts, ref_texts,
                        config.numref, config.jar_path,
                        rewrite_meteor_refs=False,
                        pre_tokenize=config.pre_tokenize,
                        norm=config.norm)
    print('METEOR:', meteor)

    bleu = get_bleu(pred_texts, ref_texts)
    print('BLEU-4:', bleu)


if __name__ == '__main__':
    main(config)
