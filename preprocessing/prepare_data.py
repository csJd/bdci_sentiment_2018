# coding: utf-8
# created by deng on 10/12/2018

import jieba
import pandas as pd
import numpy as np
from time import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from utils.path_util import from_project_root
from preprocessing.transformer import TfdcTransformer

SUBJECT_LIST = ['价格', '配置', '操控', '舒适性', '油耗', '动力', '内饰', '安全性', '空间', '外观']
N_SUBJECT = 10


def id2sub(subject_id):
    return SUBJECT_LIST[subject_id]


def sub2id(subject):
    return SUBJECT_LIST.index(subject)


def get_subject_sent(data_df, content_id, subject):
    tmp_df = data_df[data_df['content_id'] == content_id]
    if subject not in tmp_df['subject'].values:
        return 2
    return int(tmp_df[tmp_df['subject'] == subject]['sentiment_value'])


def get_content_labels(data_df, content_id, kind='subjects'):
    tmp_df = data_df[data_df['content_id'] == content_id]
    ret = list()
    for col in map(str, range(10)):
        if int(tmp_df[col]) < 2:
            ret.append(col if kind == 'subjects' else str(int(tmp_df[col])))
    return ' '.join(ret)


def gen_rematch_val():
    """ Use train data of rematch to generate gold result of test data in preliminary

    """
    train_df = pd.read_csv(from_project_root("data/train_2.csv"))
    test_df = pd.read_csv(from_project_root("data/test_public.csv"))
    val_df = test_df.merge(train_df, on='content') \
        .drop(columns=['content_id_y']) \
        .rename(columns={'content_id_x': 'content_id'})
    val_df.to_csv(from_project_root('data/test_gold.csv'), index=False)

    test_df = pd.read_csv(from_project_root("data/test_public_2.csv"))
    test_df = test_df[~test_df['content_id'].isin(val_df['content_id'])]
    test_df.to_csv('data/test.csv', index=False)


def data_to_multilabel(train_url, test_url):
    """ process data into 10 subject

        Args:
            train_url: url to train data
            test_url: url to test data

    """
    train_df = pd.read_csv(train_url)
    test_df = pd.read_csv(test_url)

    ml_df = train_df[['content_id', 'content']].drop_duplicates()
    jieba.load_userdict(from_project_root('processed_data/user_dict.txt'))
    for df in [ml_df, test_df]:
        df['content'] = df['content'].apply(str.strip)
        df['word_seg'] = df['content'].apply(lambda text: ' '.join(jieba.cut(text)))

    for subject_id, subject in enumerate(SUBJECT_LIST):
        ml_df[str(subject_id)] = ml_df['content_id'].apply(
            lambda cid: get_subject_sent(train_df, cid, subject))

    for kind in ('subjects', 'sentiment'):
        ml_df[kind] = ml_df['content_id'].apply(
            lambda cid: get_content_labels(ml_df, cid, kind))

    test_df = test_df.rename(columns={'content': 'article'})
    ml_df = ml_df.rename(columns={'content': 'article'})
    ml_df.to_csv(from_project_root("processed_data/train_ml.csv"), index=False)
    test_df.to_csv(from_project_root("processed_data/test_data.csv"), index=False)


def split_data():
    """ split data into train dta and validate data

    """
    data_df = pd.read_csv(from_project_root('processed_data/train_ml.csv'))
    train_df, val_df = train_test_split(data_df, test_size=0.1, shuffle=True, random_state=233)
    train_df.to_csv(from_project_root('processed_data/train_data.csv'), index=False)
    val_df.to_csv(from_project_root('processed_data/val_data.csv'), index=False)


def generate_vectors(train_url, test_url=None, column='article', trans_type=None, max_n=1, min_df=1, max_df=1.0,
                     max_features=1, sublinear_tf=True, balanced=False, re_weight=0, verbose=False, drop_words=0,
                     multilabel_out=False, label_col='subjects', shuffle=True):
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
        multilabel_out: return y as multilabel format
        label_col: col name of label
        shuffle: re sample train data

    Returns:
        X, y, X_test

    """
    verbose and print("loading '%s' level data from %s with pandas" % (column, train_url))

    train_df = pd.read_csv(train_url)
    if shuffle:
        train_df = train_df.sample(frac=1)

    # vectorizer
    analyzer = 'word' if column == 'word_seg' else 'char'
    vec = CountVectorizer(analyzer=analyzer, ngram_range=(1, max_n), min_df=min_df, max_df=max_df,
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
    if multilabel_out:
        y = MultiLabelBinarizer().fit_transform(train_df[label_col].apply(str.split))
    else:
        y = train_df[label_col].apply(lambda labels: int(labels.split()[0])).values
    X = trans.fit_transform(X, y)

    X_test = None
    if test_url:
        verbose and print("transforming test set")
        test_df = pd.read_csv(test_url)
        X_test = vec.transform(test_df[column])
        X_test = trans.transform(X_test)

    s_time = time()
    verbose and print("finish transforming in %.3f seconds\n" % (s_time - e_time))
    return X, y, X_test


def main():
    train_url = from_project_root("data/train_2.csv")
    test_url = from_project_root("data/test_public_2.csv")
    # data_to_multilabel(train_url, test_url)
    # split_data()
    pass


if __name__ == '__main__':
    main()
