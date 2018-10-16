# coding: utf-8
# created by deng on 2018/10/16

from time import time
from sklearn.model_selection import cross_validate
# from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.externals import joblib
from sklearn.metrics import f1_score, accuracy_score
from time import time
import numpy as np
import scipy.sparse as sp
import pandas as pd

from preprocessing.prepare_data import generate_vectors, id2sub
from utils.path_util import from_project_root
from utils.proba_util import predict_proba

N_JOBS = 6
N_CLASSES = 10
RANDOM_STATE = 10
CV = 5


def validate_clf(clf, X, y):
    """ do cross validation on clf

    Args:
        clf: clf to be tuned
        X: X for fit
        y: y for fit

    """
    s_time = time()
    scoring = 'f1_micro'
    cv_result = cross_validate(clf, X, y, cv=CV, scoring=scoring, n_jobs=N_JOBS, return_train_score=True)
    train_scores = cv_result['train_score']
    test_scores = cv_result['test_score']
    e_time = time()
    # print cv results
    print("validation is done in %.3f seconds" % (e_time - s_time))
    print("metrics of each cv: ")
    for i in range(CV):
        print(" train_f1 %f, val_f1 %f" % (train_scores[i], test_scores[i]))
    print('mean micro_f1 on val set: %f\n' % test_scores.mean())


def calc_f1(clf, X_val, threshold=0.139):
    """ validating clf according metrics described on info page

    Args:
        clf: classifier
        X_val: val data
        threshold:  threshold to be predicted

    Returns:

    """
    pred = predict_proba(clf, X_val).values.ravel()
    val = pd.read_csv(from_project_root('processed_data/val_data.csv'),
                      usecols=list(map(str, range(10)))).values.ravel()
    pred[pred < threshold] = 2
    pred[pred < 2] = 0
    pred.astype(int)
    tp, fp, fn = 0, 0, 0
    for i, j in zip(val, pred):
        if i > 1 and j > 1:
            continue
        if i > 1:
            fp += 1
        elif i == j:
            tp += 1
        else:
            fn += 1
    print(tp, fp, fn, tp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    micro_f1 = 2 * recall * precision / (recall + precision)
    print("metrics on validation set: \n recall = %f, precision = %f, micro_f1 = %f\n"
          % (recall, precision, micro_f1))


def init_clfs():
    """ init classifiers to train

    Returns:
        dict, clfs

    """
    clfs = dict()
    # clfs['xgb'] = XGBClassifier(n_jobs=-1)
    clfs['lsvc'] = LinearSVC()
    return clfs


def train_clfs(clfs, X, y, test_size=0.2, validating=False, random_state=None):
    """ train clfs

    Args:
        clfs: classifiers
        X: data X of shape (samples_num, feature_num)
        y: target y of shape (samples_num,)
        test_size: test_size for train_test_split
        validating: whether to validate classifiers
        random_state: random_state for train_test_split

    """

    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    print("train data shape", X_train.shape, y_train.shape)
    print("dev data shape  ", X_test.shape, y_test.shape)
    for clf_name in clfs:
        clf = clfs[clf_name]
        if validating:
            print("validation on %s is running" % clf_name)
            validate_clf(clf, X, y)
            continue

        print("%s model is training" % clf_name)
        if not validating:
            s_time = time()
            clf.fit(X_train, y_train)
            e_time = time()
            print(" training finished in %.3f seconds" % (e_time - s_time))

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        print(" accuracy = %f\n f1_score = %f\n" % (acc, macro_f1))

    return clfs


def validating():
    """ do validating


    """
    clfs = init_clfs()
    # load from pickle
    pk_url = from_project_root("processed_data/vector/stacked_all_subj_XyX_test_48.pk")
    print("loading data from", pk_url)
    X, y, X_val = joblib.load(pk_url)

    # train_url = from_project_root("processed_data/train_data.csv")
    # val_url = from_project_root("processed_data/val_data.csv")
    # generate from original csv
    # X, y, X_val = generate_vectors(train_url, val_url, column='article', max_n=3, min_df=2, max_df=0.8,
    #                                max_features=200000, trans_type='dc', sublinear_tf=True, balanced=False,
    #                                multilabel_out=False, label_col='subjects')

    print(X.shape, y.shape, X_val.shape)
    train_clfs(clfs, X, y, validating=True, random_state=RANDOM_STATE)
    clf = LinearSVC()
    clf.fit(X, y)
    calc_f1(clf, X_val, threshold=0.139)


def generate_result():
    train_url = from_project_root("processed_data/train_ml.csv")
    test_url = from_project_root("processed_data/test_seg.csv")
    # X, y, X_test = generate_vectors(train_url, test_url, column='article', max_n=3, min_df=2, max_df=0.8,
    #                                 max_features=200000, trans_type='dc', sublinear_tf=True, balanced=False,
    #                                 multilabel_out=False, label_col='subjects')
    X, y, X_test = joblib.load(from_project_root('processed_data/vector/stacked_all_subj_XyX_test_48.pk'))

    clf = LinearSVC()
    clf.fit(X, y)
    probas = predict_proba(clf, X_test).values
    cids = pd.read_csv(test_url, usecols=['content_id']).values.ravel()
    result_file = open(from_project_root('processed_data/result/baseline_dc_3_200000.csv'),
                       'w', newline='\n', encoding='utf-8')
    result_file.write("content_id,subject,sentiment_value,sentiment_word" + "\n")
    for i, cid in enumerate(cids):
        no_result = True
        for j in range(N_CLASSES):
            if probas[i][j] > 0.139:
                no_result = False
                out = ','.join([cid, id2sub(j), '0', '\n'])
                result_file.write(out)
        if no_result:
            result_file.write(cid + ',,,\n')
    result_file.close()


def main():
    # validating()
    generate_result()
    pass


if __name__ == '__main__':
    main()
