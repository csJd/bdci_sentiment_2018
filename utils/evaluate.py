# coding: utf-8
# created by deng on 2018/10/28

import pandas as pd
from utils.path_util import from_project_root


def evaluate(pred_url, use_senti=True):
    """ evaluate result file of preliminary test data

    Args:
        pred_url: str, url of predicted result file
        use_senti: bool, use sentiment_value column or not

    """
    usecols = ['content_id', 'subject']
    if use_senti:
        usecols.append('sentiment_value')
    true_df = pd.read_csv(from_project_root('data/preliminary/test_gold.csv'), usecols=usecols)
    pred_df = pd.read_csv(pred_url, usecols=usecols)

    # tp：判断正确的数量;
    # fp：判断错误或多判的数量;
    # fn；漏判的数量;
    tp = len(true_df.merge(pred_df, on=usecols))
    fp = len(pred_df) - tp
    fn = len(true_df) - tp
    print("metrics on test set of preliminary%s:" % ("" if use_senti else " without sentiment"))
    print(" tp = %d, fp = %d, fn = %d, n_samples = %d" % (tp, fp, fn, tp + fn))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    micro_f1 = 2 * recall * precision / (recall + precision)
    print(" recall = %f, precision = %f, micro_f1 = %f\n" % (recall, precision, micro_f1))


def main():
    evaluate(from_project_root('data/tmp.csv'))
    pass


if __name__ == '__main__':
    main()
