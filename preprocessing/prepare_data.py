# coding: utf-8
# created by deng on 10/12/2018

import jieba
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from utils.path_util import from_project_root

SUBJECT_LIST = ['价格', '配置', '操控', '舒适性', '油耗', '动力', '内饰', '安全性', '空间', '外观']
N_SUBJECT = 10


def id2sub(id):
    return SUBJECT_LIST[id]


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


def main():
    label_to_10_column()
    pass


if __name__ == '__main__':
    main()
