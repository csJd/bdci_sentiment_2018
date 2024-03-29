# coding: utf-8
# created by deng on 2018/10/16

from time import time
from sklearn.model_selection import cross_validate
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.externals import joblib
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from time import time
import numpy as np
import pandas as pd

from preprocessing.prepare_data import generate_vectors, id2sub
from utils.path_util import from_project_root
from utils.proba_util import predict_proba
from utils.evaluate import evaluate
from bow_model.classes import LinearSVCP

N_JOBS = 6
N_CLASSES = 10
RANDOM_STATE = 10


def validate_clf(clf, X, y, cv=5, scoring='f1_micro'):
    """ do cross validation on clf

    Args:
        clf: clf to be tuned
        X: X for fit
        y: y for fit
        cv: cv for cross validate
        scoring: scoring for cross validate

    """
    s_time = time()
    cv_result = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=N_JOBS, return_train_score=True)
    train_scores = cv_result['train_score']
    test_scores = cv_result['test_score']
    e_time = time()
    # print cv results
    print("validation is done in %.3f seconds" % (e_time - s_time))
    print("metrics of each cv: ")
    for i in range(cv):
        print(" train_f1 %f, val_f1 %f" % (train_scores[i], test_scores[i]))
    print('averaged %s on val set: %f\n' % (scoring, test_scores.mean()))


def calc_metrics(y_true, y_pred, y_probas=None):
    """ validating clf according metrics described in info page

    Args:
        y_true: real labels, shape = (n_samples, n_labels)
        y_pred: predicted labels, shape = (n_samples, ), (n_samples, n_labels)
        y_probas: predicted probabilities of each label, shape = (n_samples, )

    """
    if len(y_pred.shape) == 1:
        mlb = MultiLabelBinarizer().fit([range(10)])
        y_pred = mlb.transform(y_pred.reshape(-1, 1))

    if len(y_pred.shape) == 3:
        y_pred = y_pred[:, :, 1].T

    names = ['价格', '配置', '操控', '舒适', '油耗', '动力', '内饰', '安全', '空间', '外观']
    print("\ndetails before fill non result samples:\n")
    print(classification_report(y_true, y_pred, target_names=names, digits=6))

    # if y_probas gave, make sure each sample has at least 1 label
    if y_probas is not None:
        for i, row in enumerate(y_pred):
            if sum(row) < 1:
                row[y_probas[i].argmax()] = 1

    print("details after fill non result with argmax:\n")
    print(classification_report(y_true, y_pred, target_names=names, digits=6))

    # pred[pred < threshold] = 2
    # pred[pred < 2] = 0
    # tp, fp, fn = 0, 0, 0
    # for i in range(pred.shape[0]):
    #     no_result = 1
    #     for j in range(pred.shape[1]):
    #         if true[i][j] > 1 and pred[i][j] > 1:  # both == 2
    #             continue
    #         if true[i][j] == pred[i][j]:  # true == pred, correctly predicted
    #             tp += 1
    #         if pred[i][j] < 2:
    #             no_result = 0
    #             fp += 1
    #         if true[i][j] < 2:
    #             fn += 1
    #     fp += no_result
    #
    # fn -= tp
    # fp -= tp
    #
    # print(tp, fp, fn, tp + fn)
    # recall = tp / (tp + fn)
    # precision = tp / (tp + fp)
    # micro_f1 = 2 * recall * precision / (recall + precision)
    # print("metrics on validation set: \n recall = %f, precision = %f, micro_f1 = %f\n"
    #       % (recall, precision, micro_f1))


def init_clfs():
    """ init classifiers to train

    Returns:
        dict, clfs

    """
    clfs = dict()
    # clfs['xgb'] = XGBClassifier(n_jobs=-1)
    clfs['lsvc'] = LinearSVC()
    return clfs


def validate(pkl_url=None, cv=5, evaluating=False):
    """ do validating

        Args:
            pkl_url: load data from pickle file, set to None to generate data instantly
            cv: do cross validation or not
            evaluating: whether to do evaluating on test_gold

    """
    clfs = init_clfs()
    val_url = from_project_root("data/preliminary/test_gold_ex.csv")
    if pkl_url is not None:
        # load from pickle
        print("loading data from", pkl_url)
        X, y, X_val = joblib.load(pkl_url)
    else:
        train_url = from_project_root("data/preliminary/train_ex.csv")
        # generate from original csv
        X, y, X_val = generate_vectors(train_url, val_url, column='article', max_n=3, min_df=3, max_df=0.8,
                                       max_features=20000, trans_type='dc', sublinear_tf=True, balanced=True,
                                       multilabel_out=False, label_col='subjects', only_single=True, shuffle=True)

    print("data shapes:\n", X.shape, y.shape, X_val.shape)
    for name, clf in clfs.items():
        if len(y.shape) > 1:
            clf = OneVsRestClassifier(clf)
        print("cross validation on %s is running" % name)
        validate_clf(clf, X, y, cv=5, scoring='f1_micro')
        if evaluating:
            print("metrics of %s classifier:" % name)
            clf.fit(X, y)
            y_true = pd.read_csv(val_url, usecols=list(map(str, range(10)))).values < 2
            y_pred = clf.predict(X_val)
            y_probas = predict_proba(clf, X_val)
            calc_metrics(y_true, y_pred, y_probas)


def gen_10bi_result(train_url, test_url, validating=False, evaluating=False):
    """

    Args:
        train_url: url of csv train data
        test_url: url of csv  test data
        validating: whether to do validating
        evaluating: whether to do evaluating on test_gold

    Returns:
        stacked probabilities of belonging to each subjects

    """
    tdf = pd.read_csv(test_url)['content_id']
    n_samples = len(tdf)
    y_probas = np.empty(shape=(n_samples, 0))
    y_pred = np.empty(shape=(n_samples, 0), dtype=int)
    for col in range(10):
        # X, y, X_test = generate_vectors(train_url, test_url, column='article', max_n=3, min_df=3, max_df=0.8,
        #                                 max_features=30000, trans_type='dc', sublinear_tf=True, balanced=True,
        #                                 multilabel_out=False, label_col='subjects', only_single=False, shuffle=True,
        #                                 apply_fun=lambda label: str(col) in label)
        X, y, X_test = joblib.load(from_project_root("data/vector/stacked_all_XyX_val_32_%d.pk" % col))
        clf = LinearSVC()
        print("running on subject %s" % id2sub(col))
        if validating:
            validate_clf(clf, X, y, scoring='f1')
        clf.fit(X, y)
        proba = predict_proba(clf, X_test)[:, 1:2]
        y_probas = np.hstack((y_probas, proba))
        y_pred = np.hstack((y_pred, clf.predict(X_test).reshape(-1, 1)))

    if evaluating:
        y_true = pd.read_csv(test_url, usecols=list(map(str, range(10)))).values < 2
        calc_metrics(y_true, y_pred, y_probas)
    return y_pred, y_probas


def gen_multi_result(X, y, X_test):
    """ generate multilabel result use ovr classifier

    Args:
        X: (n_samples, n_features)
        y: (n_samples,) or (n_samples, n_labels)
        X_test: (n_samples, n_features)

    Returns:
        y_pred: (n_samples, n_labels)
        y_probas: (n_samples, n_labels)

    """
    clf = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
    # clf = OneVsRestClassifier(LinearSVCP())
    # clf = OneVsRestClassifier(XGBClassifier())
    if len(y.shape) == 1:
        y = MultiLabelBinarizer().fit([range(10)]).transform(y.reshape(-1, 1))
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)
    return y_pred, y_probas


def generate_result(evaluating=False, use_n_subjects='pred', senti_url=None):
    """ generate result

    Args:
        evaluating: evaluating use preliminary data
        use_n_subjects: use n_subjects info, 'gold', 'pred' or 'one'
        url to sentiment result

    """
    train_url = from_project_root("data/train_2_ex.csv")
    test_url = from_project_root("data/test_public_2v3_ex.csv")
    senti = None
    if evaluating:
        train_url = from_project_root("data/preliminary/train_ex.csv")
        test_url = from_project_root("data/preliminary/test_gold_ex.csv")
        senti = joblib.load(senti_url) if senti_url else None

    X, y, X_test = generate_vectors(train_url, test_url, column='article', max_n=3, min_df=3, max_df=0.8,
                                    max_features=20000, trans_type='hashing', sublinear_tf=True, balanced=True,
                                    multilabel_out=False, label_col='subjects', only_single=True, shuffle=True)
    X, y, X_test = joblib.load(from_project_root('data/vector/stacked_one_XyX_val_32_subjects.pk'))

    clf = LinearSVC()
    clf.fit(X, y)
    # pred, probas = clf.predict(X_test), predict_proba(clf, X_test)
    pred, probas = gen_10bi_result(train_url, test_url, validating=True, evaluating=True)
    # pred, probas = gen_multi_result(X, y, X_test)
    result_df = pd.DataFrame(columns=["content_id", "subject", "sentiment_value", "sentiment_word"])
    cids = pd.read_csv(test_url, usecols=['content_id']).values.ravel()
    for i, cid in enumerate(cids):
        k = 1
        if use_n_subjects == 'gold':
            cid_list = pd.read_csv(from_project_root('data/submit_example_2.csv'))['content_id'].tolist()
            k = cid_list.count(cid)
        elif use_n_subjects == 'pred':
            k = max(1, pred[i].sum())
        for j in probas[i].argsort()[-k:]:
            senti_val = 0 if senti is None else senti[i]
            result_df = result_df.append({'content_id': cid, 'subject': id2sub(j), 'sentiment_value': senti_val},
                                         ignore_index=True)

    save_url = from_project_root('data/result/tmp.csv')
    result_df.to_csv(save_url, index=False)
    if evaluating:
        y_true = pd.read_csv(test_url, usecols=list(map(str, range(10)))).values < 2
        calc_metrics(y_true, pred, probas)
        print("metrics on %s subjects: " % use_n_subjects)
        evaluate(save_url, use_senti=False)
        evaluate(save_url, use_senti=True)


def gen_senti_result(pkl_url):
    X, y, X_test = joblib.load(pkl_url)
    clf = XGBClassifier()
    clf.fit(X, y)
    senti = clf.predict(X_test)
    joblib.dump(senti, pkl_url.replace('.pk', '.xgb.result.pk'))


def main():
    # pkl_url = from_project_root("data/vector/stacked_one_XyX_val_32_sentiment.pk")
    # validate(pkl_url=None)
    generate_result(evaluating=True, use_n_subjects='pred')
    # gen_senti_result(pkl_url)
    pass


if __name__ == '__main__':
    main()
