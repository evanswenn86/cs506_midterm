import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer




def import_dataX(filename):
    """
    import data from specific dataset file
    then use iloc to choose the feature we need
    :param filename: String, we need the absolute file location together with file's name
    :return: X, y: pd.DataFrame, already operated according to HW0 description
    """


    data = pd.read_csv(filename)
    # print(data.columns)
    # data = data.values
    # print(data)
    data = data.dropna(subset=['Score', 'Text'])
    df = pd.DataFrame(data)
    df = drop_unnecessary(df)
    return df

def drop_unnecessary(Imported_df):
    Imported_df = Imported_df.drop(['Id', 'Summary', 'Time'],axis=1)
    return Imported_df

def tf_idf(Inported_df):
    cachedStopWords = nltk.corpus.stopwords.words("english")
    tfidf2 = TfidfVectorizer(max_df=0.7, stop_words=cachedStopWords)
    re = tfidf2.fit_transform(Inported_df['Text'])

    print(re)
    return re


# Sampled_df = import_dataX('/Users/yufanwen/PycharmProjects/cs506_midterm/bu-cs-506-fall-2019-midterm-competition/sample.csv')
imported_df = import_dataX('/Users/yufanwen/PycharmProjects/cs506_midterm/bu-cs-506-fall-2019-midterm-competition/train.csv')
# print(Sampled_df)
tf_idf(imported_df)

