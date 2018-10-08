import codecs
import json
import os
import sys

import numpy as np
import scipy as sp
import scipy.stats


def walkdir(rootdir, extension='.txt', exact_match=False):
    if exact_match:
        p = os.path.join(rootdir, extension)
        return [p] if os.path.exists(p) else []

    filelist = []
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.endswith(extension):
                filelist.append(os.path.join(parent, filename))
    return filelist


def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def str2bool(s):
    return s.lower() not in ['false', 'f', '0', 'none', 'no', 'n']


class ConfigParser(object):
    parsed_args = []

    def __init__(self, args=[], prefix='', check=True):
        super(ConfigParser, self).__init__()
        self._prefix = prefix
        self._sep = '/'
        self._prefixed_attrs = list()
        self._parse_args(args, check=check)

    def _parse_args(self, args=[], check=True):

        for i, arg in enumerate(args):
            if arg[:2] != '--':
                continue
            try:
                tmp_str = arg[2:]
                idx = tmp_str.index('=')
                k, v = tmp_str[:idx], tmp_str[(idx + 1):]

                levels = k.split(self._sep)
                assert len(levels) <= 2
                if len(levels) == 1:
                    k = levels[0]
                    if k in self._prefixed_attrs:
                        continue
                elif levels[0] != self._prefix:
                    continue
                else:
                    k = levels[1]
                    self._prefixed_attrs.append(k)

                if k in self.__dict__:
                    self.parsed_args.append(arg)
                    type_of_attr = type(self.__dict__[k])
                    if type_of_attr == bool:
                        self.__dict__[k] = str2bool(v)
                    elif type_of_attr == type:
                        self.__dict__[k] = eval(v)
                    else:
                        self.__dict__[k] = type_of_attr(v)
            except Exception as e:
                print('Bad argument:', e)
                sys.exit()

        if check:
            self._check()

    def _check(self):
        pass

    def __str__(self):
        return json.dumps(self.__dict__)


def save_configs(configs, names, path='config.json', msg=''):
    configs = [str(c) for c in configs]
    out = ',\n'.join('"{}": {}'.format(n, c) for n, c in zip(names, configs))

    if msg:
        out += ',\n' + msg

    with codecs.open(path, mode='w', encoding='utf-8') as f:
        f.write('{\n' + out + '\n}\n')


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    # Or:
    # scipy.stats.t.interval(0.95, n - 1,
    #                        loc=np.mean(a),
    #                        scale=scipy.stats.sem(a))
    return m, h


def unify_len_for_list_of_list(lst2, pad_value=0, max_len=None):
    max_len = max_len or len(max(lst2, key=len))
    ret = [l[:max_len] + [pad_value] * (max_len - len(l)) for l in lst2]
    return ret


def sort_lists(lsts, key, reverse=False):
    n_lsts = len(lsts)
    if n_lsts == 1:
        return [x for _, x in sorted(zip(key, *lsts), reverse=reverse)]
    elif n_lsts == 2:
        lsts = [(x, y) for _, x, y in sorted(zip(key, *lsts), reverse=reverse)]
        return list(zip(*lsts))
    elif n_lsts == 3:
        lsts = [(x, y, z) for _, x, y, z in sorted(zip(key, *lsts),
                                                   reverse=reverse)]
        return list(zip(*lsts))
    else:
        raise NotImplementedError()


def split_arr_by_diag(arr):
    shape = arr.shape
    assert len(shape) == 2

    diag = np.diag(arr)

    n, m = shape
    sel = np.logical_not(np.eye(n, m, dtype=bool))
    non_diag = arr[sel].reshape(n, m - 1)
    return diag, non_diag


def pair_kl_divergence(mu1, sd1, mu2, sd2):
    """Pair-wise kl divergence in Numpy.

    Args:
        mu1, sd1: [n, d]
        mu2, sd2: [m, d]

    Returns:
        div: [n, m]
    """
    import functools
    import operator

    if type(mu1) is not np.ndarray:
        mu1 = np.array(mu1)
    if type(sd1) is not np.ndarray:
        sd1 = np.array(sd1)
    if type(mu2) is not np.ndarray:
        mu2 = np.array(mu2)
    if type(sd2) is not np.ndarray:
        sd2 = np.array(sd2)

    _, d = mu1.shape

    var1 = sd1 ** 2
    var2 = sd2 ** 2

    mu1 = np.expand_dims(mu1, 1)           # [n, 1, d]
    var1 = np.expand_dims(var1, 1)        # [n, 1, d]

    # Calc: norm_dmu = (u1 - u2)^T * Sigma_2^{-1} * (u1 - u2)
    # ((([n, 1, d] - [m, d]) ** 2) * (1. / [m, d]))
    norm_dmu = (((mu1 - mu2) ** 2) * (1. / var2)).sum(axis=-1)

    # Calc: tr = trace(Sigma_2^{-1} * Sigma_1)
    # ([n, 1, d] / [m, d]).sum() -> [n, m]
    tr = (var1 / var2).sum(axis=-1)

    # Cacl: deter = ln(det(sd_1) / det(sd_2))
    def get_deter(sd):  # [n, d]
        det = np.apply_along_axis(
            lambda a: functools.reduce(operator.mul, a), 1, sd)
        return det

    deter1 = get_deter(sd1)                 # [n]
    deter2 = get_deter(sd2)                 # [m]
    deter1 = np.expand_dims(deter1, 1)      # [n, 1]
    deter = np.log(deter1 / deter2)         # [n, m]

    div = (norm_dmu + tr - d) / 2 - deter

    return div


def get_cur_git_rev():
    import subprocess
    out = subprocess.check_output('git rev-parse HEAD'.split())
    rev = out.decode()
    rev = rev.strip()
    return rev


class LRAdjust(object):
    def __init__(self, initial_lr, decay_rate, thresh, width):
        self._initial_lr = initial_lr
        self._thresh = thresh
        self._decay_rate = decay_rate
        self._width = width

        assert thresh > 0

        print('initial_lr:', initial_lr)
        print('thresh:', thresh)
        print('decay_rate:', decay_rate)
        print('width:', width)

        self._last_score = None
        self._window = []  # record the progress in a range of steps

    def adjust(self, i, lr, score):
        if self._last_score is None:
            self._last_score = score
            return lr

        # progress is defined as relative improvement of score
        progress = abs(score - self._last_score) / abs(self._last_score)
        self._window.append(progress)

        if len(self._window) < self._width:
            # if the window is not full, stay.
            new_lr = lr
        else:
            # make decision when window is full
            assert len(self._window) == self._width
            if all(progress < self._thresh for progress in self._window):
                # if progress is below the threshold for every step in the
                # window, we need to decrease the learning rate.
                new_lr = lr * self._decay_rate
                self._window = []
            else:
                # otherwise, we shall wait
                new_lr = lr
                self._window = self._window[1:]

        self._last_score = score
        return new_lr
