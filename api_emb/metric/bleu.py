from . import codenn as cb


def get_bleu(pred_texts, ref_texts):
    """Compute BLEU score.

    pred_texts: [n], a list of n texts.
    ref_texts: [n, k], a list of list. ref_texts[i] is a list of k references.
    """
    score = [0.] * 5
    num = len(pred_texts)

    assert len(pred_texts) == len(ref_texts)

    for refs, pred in zip(ref_texts, pred_texts):
        refs = [ref.lower() for ref in refs]
        pred = pred.lower()
        bl = cb.bleu(refs, pred)
        score = [score[i] + bl[i] for i in range(0, len(bl))]

    return [s / num * 100.0 for s in score]
