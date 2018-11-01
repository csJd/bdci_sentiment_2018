# coding: utf-8
# created by deng on 8/21/2018

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.calibration import CalibratedClassifierCV
from os.path import basename

from utils.path_util import from_project_root

N_CLASSES = 10


def pk_to_csv(pk_url):
    """
    Args:
        pk_url: url to pk proba file

    """
    save_url = pk_url.replace('pk', 'csv')
    arr = joblib.load(pk_url)
    proba_df = pd.DataFrame(arr, columns=['class_prob_' + str(i + 1) for i in range(N_CLASSES)])
    proba_df.to_csv(save_url, index_label='id')
    return proba_df


def predict_proba(clf, X_test, X=None, y=None, save_url=None, n_classes=None):
    """ train clf and get proba predict

    Args:
        clf: trained classifier
        X: X for fit
        y: y for fit
        X_test: X_test for predict
        save_url: url to save result, not save if set it to None
        n_classes: num of classes

    Returns:
        DataFrame: proba_df

    """
    try:
        proba = clf.predict_proba(X_test)
    except AttributeError:
        try:
            proba = clf._predict_proba_lr(X_test)
        except AttributeError:
            if X is None or y is None:
                print("X and y is required for CalibratedClassifierCV")
                return
            clf = CalibratedClassifierCV(clf)
            clf.fit(X, y)
            proba = clf.predict_proba(X_test)

    if n_classes is None:
        return proba

    proba_df = pd.DataFrame(proba, columns=['class_prob_' + str(i + 1) for i in range(N_CLASSES)])
    if save_url is None:
        pass
    elif save_url.endswith('.pk'):
        joblib.dump(proba_df, save_url)
    elif save_url.endswith('.csv'):
        proba_df.to_csv(save_url, index_label='id')
    return proba_df


def merge_probas(proba_dict, save_url):
    """  merge proba results

    Args:
        proba_dict: dict, {url: weight}

    """
    sum_df = None
    for url in proba_dict:
        print("{}:{};".format(basename(url), proba_dict[url]))
        proba_df = pd.read_csv(url, index_col='id') if url.endswith('.csv') else pk_to_csv(url)
        if sum_df is None:
            sum_df = proba_df * proba_dict[url]
        else:
            sum_df += proba_df * proba_dict[url]
    result_df = pd.DataFrame(np.argmax(sum_df.values, axis=1) + 1, columns=['class'])
    result_df.to_csv(save_url, index_label='id')


def main():
    proba_dict = {
        from_project_root('processed_data/result/result1.csv'): 0.1,
        from_project_root('processed_data/result/result2.csv'): 0.9,
    }
    save_url = from_project_root('processed_data/result.csv')
    merge_probas(proba_dict, save_url)
    pass


if __name__ == '__main__':
    main()
