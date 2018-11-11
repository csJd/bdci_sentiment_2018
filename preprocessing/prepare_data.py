# coding: utf-8
# created by deng on 10/12/2018

import jieba
import pandas as pd
import numpy as np
from time import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
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
    test_df = pd.read_csv(from_project_root("data/preliminary/test_public.csv"))
    val_df = test_df.merge(train_df, on='content') \
        .drop(columns=['content_id_y']) \
        .rename(columns={'content_id_x': 'content_id'})
    val_df.to_csv(from_project_root('data/preliminary/test_gold.csv'), index=False)

    test_df = pd.read_csv(from_project_root("data/test_public_2.csv"))
    test_df = test_df[~test_df['content_id'].isin(val_df['content_id'])]
    test_df.to_csv('data/test_2.csv', index=False)


def extend_data(data_url, for_sentiment=True):
    """ generate more information into data file

        Args:
            data_url: url to original data file
            for_sentiment: process for sentiment

    """
    df_orig = pd.read_csv(data_url)

    df = df_orig[['content_id', 'content']].drop_duplicates() if not for_sentiment else df_orig
    jieba.load_userdict(from_project_root('data/processed/user_dict.txt'))

    if for_sentiment:
        df['content'] = df['content'] + df['subject']
    df['content'] = df['content'].apply(str.strip)
    df['word_seg'] = df['content'].apply(lambda text: ' '.join(jieba.cut(text)))
    df['cut_all'] = df['content'].apply(lambda text: ' '.join(jieba.cut(text, cut_all=True)))
    df['cut_for_search'] = df['content'].apply(lambda text: ' '.join(jieba.cut_for_search(text)))

    if 'subject' in df_orig and not for_sentiment:
        for subject_id, subject in enumerate(SUBJECT_LIST):
            df[str(subject_id)] = df['content_id'].apply(
                lambda cid: get_subject_sent(df_orig, cid, subject))

        for kind in ('subjects', 'sentiment'):
            df[kind] = df['content_id'].apply(
                lambda cid: get_content_labels(df, cid, kind))

        df['n_subjects'] = df['subjects'].apply(lambda subs: len(subs.split()))

    df = df.rename(columns={'content': 'article'})
    new_suffix = '_exs.csv' if for_sentiment else '_ex.csv'
    save_url = data_url.replace('.csv', new_suffix)
    df.to_csv(save_url, index=False)


def split_data(data_url, test_size=0.2):
    """ split data into train dta and validate data

    Args:
        data_url: str, url of data file to be split
        test_size: float, test_size for train_test_split

    """
    data_df = pd.read_csv(data_url)
    train_df, val_df = train_test_split(data_df, test_size=test_size, shuffle=True, random_state=233)
    train_df.to_csv(data_url.replace('.csv', '_train.csv'), index=False)
    val_df.to_csv(data_url.replace('.csv', '_val.csv'), index=False)


def generate_vectors(train_url, test_url=None, column='article', trans_type=None, max_n=1, min_df=1, max_df=1.0,
                     max_features=1, sublinear_tf=True, balanced=False, re_weight=0, verbose=False, drop_words=0,
                     multilabel_out=False, label_col='subjects', only_single=True, shuffle=True,
                     apply_fun=None):
    """ generate X, y, X_test vectors with csv(with header) url use pandas and CountVectorizer

    Args:
        train_url: url to train csv
        test_url: url to test csv, set to None if not need X_test
        column: column to use as feature
        trans_type: specific transformer, {'dc','idf', 'hashing'}
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
        only_single: only keep records of single label
        shuffle: re sample train data
        apply_fun: callable to be applied on label column

    Returns:
        X, y, X_test

    """
    verbose and print("loading '%s' level data from %s with pandas" % (column, train_url))

    train_df = pd.read_csv(train_url)
    if shuffle:
        train_df = train_df.sample(frac=1)
    if only_single:
        train_df = train_df[train_df['subjects'].apply(lambda x: len(x) < 2)]

    # vectorizer
    s_time = time()
    analyzer = 'word' if column == 'word_seg' else 'char'
    vec = CountVectorizer(analyzer=analyzer, ngram_range=(1, max_n), min_df=min_df, max_df=max_df,
                          max_features=max_features, token_pattern='\w+')
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
    X = sequences if trans_type == 'hashing' else vec.fit_transform(sequences)
    e_time = time()
    verbose and print("finish vectorizing in %.3f seconds, transforming" % (e_time - s_time))

    # transformer
    if trans_type is None or trans_type == 'idf':
        trans = TfidfTransformer(sublinear_tf=sublinear_tf, use_idf=balanced)
    elif trans_type == 'dc':
        trans = TfdcTransformer(sublinear_tf=sublinear_tf, balanced=balanced, re_weight=re_weight)
    else:
        trans = HashingVectorizer(analyzer=analyzer, ngram_range=(1, max_n), n_features=max_features,
                                  token_pattern='\w+', binary=not balanced)
    verbose and print(trans_type, "transformer params:", trans.get_params())

    if multilabel_out:
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(train_df[label_col].apply(str.split))
        verbose and print("multilabel columns:\n", mlb.classes_)
    else:
        y = train_df[label_col].apply(apply_fun).values if apply_fun is not None \
            else train_df[label_col].values
    X = trans.fit_transform(X, y)

    X_test = None
    if test_url:
        verbose and print("transforming test set")
        test_df = pd.read_csv(test_url)
        X_test = test_df[column] if trans_type == 'hashing' else vec.transform(test_df[column])
        X_test = trans.transform(X_test)
    s_time = time()
    verbose and print("finish transforming in %.3f seconds\n" % (s_time - e_time))
    return X, y, X_test


def main():
    data_url = from_project_root("data/preliminary/best_subject.csv")
    extend_data(data_url, for_sentiment=True)
    pass


if __name__ == '__main__':
    main()
