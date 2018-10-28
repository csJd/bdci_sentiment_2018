# coding: utf-8
# created by deng on 2018/10/28

import pandas as pd
from utils.path_util import from_project_root


def evaluate(pred_url):
    """ evaluate result file of preliminary test data

    Args:
        pred_url: url of predicted result file

    """
    usecols = ['content_id', 'subject', 'sentiment_value']
    true_df = pd.read_csv(from_project_root('data/test_gold.csv'), usecols=usecols)
    pred_df = pd.read_csv(pred_url, usecols=usecols)

    # tp：判断正确的数量;
    # fp：判断错误或多判的数量;
    # fn；漏判的数量;
    tp = len(true_df.merge(pred_df, on=usecols))
    fp = len(pred_df) - tp
    fn = len(true_df) - tp

    print(tp, fp, fn, tp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    micro_f1 = 2 * recall * precision / (recall + precision)
    print("metrics on evaluation set: \n recall = %f, precision = %f, micro_f1 = %f\n"
          % (recall, precision, micro_f1))


def main():
    evaluate(from_project_root('data/0.6485_pre_b.csv'))
    pass


if __name__ == '__main__':
    main()
