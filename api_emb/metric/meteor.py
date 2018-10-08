"""Calculate METEOR score.

See: https://paste.ubuntu.com/26405875/
https://www.cs.cmu.edu/~alavie/METEOR/index.html
"""
import os
import tempfile

from .codenn import normalize


def get_meteor(pred_texts, ref_texts, numref, meteor_jar_path,
               rewrite_meteor_refs=False, pre_tokenize=True, norm=False):
    assert len(pred_texts) == len(ref_texts)

    (fd1, filename1) = tempfile.mkstemp()
    (fd2, filename2) = tempfile.mkstemp()
    f1 = os.fdopen(fd1, "w")
    f2 = os.fdopen(fd2, "w")
    for refs, pred in zip(ref_texts, pred_texts):
        for ref in refs:
            if pre_tokenize:
                ref = ' '.join(normalize(ref, lower=True))
            f1.write(ref + '\n')
        if pre_tokenize:
            pred = ' '.join(normalize(pred, lower=True))
        f2.write(pred + '\n')
    f1.close()
    f2.close()

    command = ('java', '-Xmx2G', '-jar', meteor_jar_path,
               filename2, filename1, '-noPunct', '-r', str(numref),
               '-norm' if norm else '')
    op = os.popen(' '.join(command)).read()
    os.remove(filename1)
    os.remove(filename2)
    return float(op.split('\n')[-2].split()[-1]) * 100
