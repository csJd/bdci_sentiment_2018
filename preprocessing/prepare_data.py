# coding: utf-8
# created by deng on 10/12/2018

import jieba
import pandas as pd
import numpy as np
from time import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from utils.path_util import from_project_root
from preprocessing.transformer import TfdcTransformer

SUBJECT_LIST = ['价格', '配置', '操控', '舒适性', '油耗', '动力', '内饰', '安全性', '空间', '外观']
N_SUBJECT = 10


def id2sub(subject_id):
    return SUBJECT_LIST[subject_id]


def get_subject_sent(data_df, content_id, subject):
    tmp_df = data_df[data_df['content_id'] == content_id]
    if subject not in tmp_df['subject'].values:
        return 2
    return int(tmp_df[tmp_df['subject'] == subject]['sentiment_value'])


def label_to_10_column():
    """ process data into 10 subject
    """
    train_df = pd.read_csv(from_project_root("data/train.csv"))
    test_df = pd.read_csv(from_project_root("data/test_public.csv"))

    new_df = train_df[['content_id', 'content']].drop_duplicates()
    for df in [new_df, test_df]:
        df['content'] = df['content'].apply(lambda text: text.replace(' ', ''))
        df['word_seg'] = df['content'].apply(lambda text: ' '.join(jieba.cut(text)))
        df['content'] = df['content'].apply(lambda text: ' '.join(text))

    for subject_id, subject in enumerate(SUBJECT_LIST):
        new_df[str(subject_id)] = new_df['content_id'].apply(
            lambda cid: get_subject_sent(train_df, cid, subject))

    test_df = test_df.rename(columns={'content': 'article'})
    new_df = new_df.rename(columns={'content': 'article'})
    new_df.to_csv(from_project_root("data/train_10.csv"), index=False)
    test_df.to_csv(from_project_root("data/test_cut.csv"), index=False)


def generate_vectors(train_url, test_url=None, column='article', trans_type=None, max_n=1, min_df=1, max_df=1.0,
                     max_features=1, sublinear_tf=True, balanced=False, re_weight=0, verbose=False, drop_words=0):
    """ generate X, y, X_test vectors with csv(with header) url use pandas and CountVectorizer

    Args:
        train_url: url to train csv
        test_url: url to test csv, set to None if not need X_test
        column: column to use as feature
        trans_type: specific transformer, {'dc','idf'}
        max_n: max_n for ngram_range
        min_df: min_df for CountVectorizer
        max_df: max_df for CountVectorizer
        max_features: max_features for CountVectorizer
        sublinear_tf: sublinear_tf for default TfdcTransformer
        balanced: balanced for default TfdcTransformer, for idf transformer, it is use_idf
        re_weight: re_weight for TfdcTransformer
        verbose: True to show more information
        drop_words: randomly delete some words from sentences

    Returns:
        X, y, X_test

    """
    verbose and print("loading '%s' level data from %s with pandas" % (column, train_url))

    train_df = None  # load_to_df(train_url)

    # vectorizer
    vec = CountVectorizer(ngram_range=(1, max_n), min_df=min_df, max_df=max_df,
                          max_features=max_features, token_pattern='\w+')
    s_time = time()
    verbose and print("finish loading, vectorizing")
    verbose and print("vectorizer params:", vec.get_params())

    sequences = train_df[column]
    # delete some words randomly
    for i, row in enumerate(sequences):
        if drop_words <= 0:
            break
        if np.random.ranf() < drop_words:
            row = np.array(row.split())
            sequences.at[i] = ' '.join(row[np.random.ranf(row.shape) > 0.35])

    X = vec.fit_transform(sequences)
    e_time = time()
    verbose and print("finish vectorizing in %.3f seconds, transforming" % (e_time - s_time))
    # transformer
    if trans_type is None or trans_type == 'idf':
        trans = TfidfTransformer(sublinear_tf=sublinear_tf, use_idf=balanced)
    else:
        trans = TfdcTransformer(sublinear_tf=sublinear_tf, balanced=balanced, re_weight=re_weight)

    verbose and print("transformer params:", trans.get_params())
    y = np.array((train_df["class"]).astype(int))
    X = trans.fit_transform(X, y)

    X_test = None
    if test_url:
        verbose and print("transforming test set")
        test_df = None  # load_to_df(test_url)
        X_test = vec.transform(test_df[column])
        X_test = trans.transform(X_test)

    s_time = time()
    verbose and print("finish transforming in %.3f seconds\n" % (s_time - e_time))
    return X, y, X_test


def main():
    # label_to_10_column()
    pass


if __name__ == '__main__':
    main()
