import logging


class Metric(object):
    """Metric."""

    def __init__(self, m, h=None, large_is_better=True, confidence=0.95):
        super(Metric, self).__init__()
        assert h is None or h >= 0.
        self.m = m
        self.h = h
        self.large_is_better = large_is_better
        self.confidence = confidence

        if h is None:
            self.l = m
            self.u = m
        else:
            self.l = m - h
            self.u = m + h

    def __str__(self):
        if self.h is None:
            if type(self.m) in {list, tuple}:
                return '[' + ', '.join('{:.4f}'.format(x) for x in self.m) + ']'
            else:
                return '{:.4f}'.format(self.m)
        else:
            return '{:.4f} ({:.6f})'.format(self.m, self.h)

    def __format__(self, format_spec):
        return self.__str__()

    def __lt__(self, other):
        if self.large_is_better:
            return self.m < other.m
        else:
            return self.m > other.m

    def __eq__(self, other):
        def _close(x, y):
            return abs(x - y) < 1e-8
        return _close(self.m, other.m) and _close(self.h, other.h)


class Perf(object):
    """Perf."""

    def __init__(self, name='perf', large_is_better=True):
        super(Perf, self).__init__()
        self.name = name
        self.large_is_better = large_is_better

        self.cur_valid = None
        self.cur_test = None

        self.best_valid = None
        self.best_valid_updated = False
        self.test_when_best_valid = None
        self.best_test = None
        self.best_test_updated = False

        # Keep when achieve best.
        self.valid_cnt = 0
        self.best_valid_at = 0
        self.test_cnt = 0
        self.best_test_at = 0

    def better(self, a, b):
        if a is None:
            return False
        if b is None:
            return True
        if self.large_is_better:
            return a > b
        else:
            return a < b

    def _log(self, msg, prev_best, cur_best, at, updated):
        extra = '(updated)' if updated else ''
        if prev_best is None:
            logging.info('{} {}: None --> {:.4f} (at {}) {}'
                         .format(msg, self.name, cur_best, at, extra))
        else:
            logging.info('{} {}: {:.4f} --> {:.4f} (at {}) {}'
                         .format(msg, self.name, prev_best,
                                 cur_best, at, extra))

    def update(self, cur, mode='valid', log=True):
        assert mode in {'valid', 'test'}

        if mode == 'valid':
            self.cur_valid = cur
            self.best_valid_updated = False
            prev_valid = self.best_valid
            if self.better(cur, self.best_valid):
                self.best_valid = cur
                self.best_valid_updated = True
                self.best_valid_at = self.valid_cnt
            if log:
                self._log('Best valid', prev_valid,
                          self.best_valid, self.best_valid_at,
                          self.best_valid_updated)
            self.valid_cnt += 1
        else:
            self.cur_test = cur
            self.best_test_updated = False
            prev_test = self.best_test
            prev_test_when_best_valid = self.test_when_best_valid

            if self.better(cur, self.best_test):
                self.best_test = cur
                self.best_test_updated = True
                self.best_test_at = self.test_cnt

            if self.best_valid_updated:
                self.test_when_best_valid = cur

            if log:
                if log:
                    self._log('Best test',
                              prev_test,
                              self.best_test,
                              self.best_test_at,
                              self.best_test_updated)
                    self._log('Test when get best valid',
                              prev_test_when_best_valid,
                              self.test_when_best_valid,
                              self.best_valid_at,
                              self.best_valid_updated)

            self.test_cnt += 1

    def show(self, fmt='simple'):
        if fmt == 'simple':
            logging.info('{:.4f} (at {}) {:.4f} {:.4f} (at {})'
                         .format(self.best_valid, self.best_valid_at,
                                 self.test_when_best_valid,
                                 self.best_test, self.best_test_at))
        else:
            raise NotImplementedError()
