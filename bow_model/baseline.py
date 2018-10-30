# coding: utf-8
# created by deng on 7/31/2018

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC

from utils.path_util import from_project_root
from preprocessing.prepare_data import id2sub


def tfidf_baseline():
    column = "word_seg"
    train_df = pd.read_csv(from_project_root('data/train_10.csv'))
    test_df = pd.read_csv(from_project_root('data/test_cut.csv'))
    vec = TfidfVectorizer(ngram_range=(1, 4), min_df=2, max_df=0.8, max_features=200000, sublinear_tf=1)
    X = vec.fit_transform(train_df[column])
    X_test = vec.transform(test_df[column])

    for i in range(10):
        y = (train_df[str(i)]).astype(int)
        clf = LinearSVC()
        clf.fit(X, y)
        test_df[str(i)] = clf.predict(X_test)
    test_df.to_csv(from_project_root('processed_data/tmp.csv'))

    result_file = open(from_project_root('processed_data/result/baseline_idf_4_200000_1.csv'),
                       'w', newline='\n', encoding='utf-8')
    result_file.write("content_id,subject,sentiment_value,sentiment_word" + "\n")
    for index, row in test_df.iterrows():
        no_result = True
        for i in range(10):
            if row[str(i)] < 2:
                no_result = False
                out = ','.join([row['content_id'], id2sub(i), str(row[str(i)]), '\n'])
                result_file.write(out)
        if no_result:
            result_file.write(row['content_id'] + ',动力,0,\n')
    result_file.close()


def main():
    tfidf_baseline()
    pass


if __name__ == '__main__':
    main()
