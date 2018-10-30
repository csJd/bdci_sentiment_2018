# coding: utf-8
# created by deng on 2018/10/16

from time import time
from sklearn.model_selection import cross_validate
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.externals import joblib
from sklearn.metrics import f1_score, accuracy_score
from time import time
import numpy as np
import pandas as pd

from preprocessing.prepare_data import generate_vectors, id2sub
from utils.path_util import from_project_root
from utils.proba_util import predict_proba
from utils.evaluate import evaluate

N_JOBS = 6
N_CLASSES = 10
RANDOM_STATE = 10
CV = 5


def validate_clf(clf, X, y, scoring='f1_micro'):
    """ do cross validation on clf

    Args:
        clf: clf to be tuned
        X: X for fit
        y: y for fit
        scoring: scoring for cross validate

    """
    s_time = time()
    cv_result = cross_validate(clf, X, y, cv=CV, scoring=scoring, n_jobs=N_JOBS, return_train_score=True)
    train_scores = cv_result['train_score']
    test_scores = cv_result['test_score']
    e_time = time()
    # print cv results
    print("validation is done in %.3f seconds" % (e_time - s_time))
    print("metrics of each cv: ")
    for i in range(CV):
        print(" train_f1 %f, val_f1 %f" % (train_scores[i], test_scores[i]))
    print('averaged %s on val set: %f\n' % (scoring, test_scores.mean()))


def calc_f1(clf, X_val, val_url, senti=None, threshold=0.139):
    """ validating clf according metrics described in info page

    Args:
        clf: classifier
        X_val: val data
        val_url: url to validation csv file
        senti: sentiment of each content in val data
        threshold:  threshold to be predicted as subject

    """
    pred = predict_proba(clf, X_val).values
    true = pd.read_csv(val_url, usecols=list(map(str, range(10)))).values
    pred[pred < threshold] = 2
    pred[pred < 2] = 0
    tp, fp, fn = 0, 0, 0
    if senti is None:
        senti = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        no_result = 1
        for j in range(pred.shape[1]):
            if true[i][j] > 1 and pred[i][j] > 1:  # both == 2
                continue
            if true[i][j] == pred[i][j]:  # true == pred, correctly predicted
                tp += 1
            if pred[i][j] < 2:
                no_result = 0
                fp += 1
            if true[i][j] < 2:
                fn += 1
        fp += no_result

    fn -= tp
    fp -= tp

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

        else:
            print("%s model is training" % clf_name)
            s_time = time()
            clf.fit(X_train, y_train)
            e_time = time()
            print(" training finished in %.3f seconds" % (e_time - s_time))

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='micro')
        print(" accuracy = %f\n f1_score = %f\n" % (acc, macro_f1))

    return clfs


def validate(pkl_url=None):
    """ do validating

        Args:
            pkl_url: load data from pickle file, set to None to generate data instantly

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

    print(X.shape, y.shape, X_val.shape)
    train_clfs(clfs, X, y, validating=True, random_state=RANDOM_STATE)
    for name in clfs:
        print("metrics of %s classifier:" % name)
        clfs[name].fit(X, y)
        calc_f1(clfs[name], X_val, val_url=val_url, senti=None, threshold=0.139)


def gen_10bi_probas(train_url, test_url, validating=False):
    """

    Args:
        train_url: url of csv train data
        test_url: url of csv  test data
        validating: whether to do validating

    Returns:
        stacked probabilities of belonging to each subjects

    """
    tdf = pd.read_csv(test_url)['content_id']
    n_samples = len(tdf)
    probas = np.empty(shape=(n_samples, 0))
    for col in range(10):
        X, y, X_test = generate_vectors(train_url, test_url, column='article', max_n=3, min_df=2, max_df=0.8,
                                        max_features=200000, trans_type='dc', sublinear_tf=True, balanced=True,
                                        multilabel_out=False, label_col=str(col), only_single=False, shuffle=True,
                                        apply_fun=lambda label: int(label < 2))
        clf = LinearSVC()
        if validating:
            print("validating on subject %s:" % id2sub(col))
            validate_clf(clf, X, y, scoring='f1_micro')
        clf.fit(X, y)
        proba = predict_proba(clf, X_test)[:, 1:2]
        probas = np.hstack((probas, proba))
    print(probas[:3, :])
    return probas


def generate_result(evaluating=False, use_n_subjects=False):
    """

    Args:
        evaluating: evaluating use preliminary data
        use_n_subjects: use n_subjects info

    Returns:

    """
    cid_list = pd.read_csv(from_project_root('data/submit_example_2.csv'))['content_id'].tolist()
    train_url = from_project_root("data/train_2_ex.csv")
    test_url = from_project_root("data/test_public_2v3_ex.csv")
    if evaluating:
        train_url = from_project_root("data/preliminary/train_ex.csv")
        test_url = from_project_root("data/preliminary/test_public_ex.csv")

    X, y, X_test = generate_vectors(train_url, test_url, column='article', max_n=3, min_df=3, max_df=0.8,
                                    max_features=20000, trans_type='dc', sublinear_tf=True, balanced=True,
                                    multilabel_out=False, label_col='subjects', only_single=True, shuffle=True)
    # X, y, X_test = joblib.load(from_project_root('processed_data/vector/stacked_all_XyX_test_48_subjects.pk'))

    # clf = LinearSVC()
    # clf.fit(X, y)
    # probas = predict_proba(clf, X_test)
    probas = gen_10bi_probas(train_url, test_url, validating=True)
    cids = pd.read_csv(test_url, usecols=['content_id']).values.ravel()
    result_df = pd.DataFrame(columns=["content_id", "subject", "sentiment_value", "sentiment_word"])

    for i, cid in enumerate(cids):
        k = cid_list.count(cid) if use_n_subjects else 1
        for j in probas[i].argsort()[-k:]:
            result_df = result_df.append({'content_id': cid, 'subject': id2sub(j), 'sentiment_value': '0'},
                                         ignore_index=True)
        if k == 0:
            result_df = result_df.append({'content_id': cid}, ignore_index=True)

    save_url = from_project_root('data/result/bdc_0.6277.csv')
    result_df.to_csv(save_url, index=False)
    if evaluating:
        evaluate(save_url, use_senti=False)
        evaluate(save_url, use_senti=True)


def main():
    # validate()
    generate_result(evaluating=True)
    pass


if __name__ == '__main__':
    main()
