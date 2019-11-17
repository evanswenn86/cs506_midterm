import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def import_dataX(filename_train, filename_test):
    data_x = pd.read_csv(filename_train)
    # df_x = pd.DataFrame(data_x)

    data_y = pd.read_csv(filename_test)
    data_y = pd.merge(data_x, data_y, how='inner', on='Id')
    data_x = data_x.dropna(subset=['Score', 'Text'])
    data_x = drop_unnecessary(data_x)
    data_x = data_x.drop(['Id'], axis=1)

    data_y = data_y.dropna(subset=['Text'])
    data_y = drop_unnecessary(data_y)
    data_y = data_y.drop(['Score_x'], axis=1)

    #     print(data_y)
    return data_x, data_y


def drop_unnecessary(Imported_df):
    Imported_df = Imported_df.drop(['Summary',
                                    'Time',
                                    'ProductId',
                                    'UserId',
                                    'HelpfulnessNumerator',
                                    'HelpfulnessDenominator'], axis=1)
    # print(Imported_df)
    return Imported_df


def extract_keywords(Inported_df):
    keyword_list = []
    cachedStopWords = nltk.corpus.stopwords.words("english")
    cv = CountVectorizer(max_df=0.9, stop_words=cachedStopWords, max_features=10000)
    word_count_vector = cv.fit_transform(Inported_df['Text'])
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    feature_names = cv.get_feature_names()

    for text in Inported_df['Text']:
        tf_idf_vector = tfidf_transformer.transform(cv.transform([text]))

        sorted_idx = sort_coo(tf_idf_vector.tocoo())
        keywords = extract_topn_from_vector(feature_names, sorted_idx, 20)
        temp = []
        for k in keywords:
            temp.append(k)

        string = " ".join(temp)
        keyword_list.append(string)

    keyword_series = pd.Series(keyword_list)
    keyword_series.rename("Keyword")

    Inported_df["Keyword"] = keyword_list
    Inported_df = Inported_df.drop(['Text'], axis=1)
    Inported_df = split_keywords(Inported_df)
    # print(Inported_df)
    return Inported_df


def split_keywords(df):
    df['kw1'], \
    df['kw2'], \
    df['kw3'], \
    df['kw4'], \
    df['kw5'], \
    df['kw6'], \
    df['kw7'], \
    df['kw8'], \
    df['kw9'], \
    df['kw10'], \
    df['kw11'], \
    df['kw12'], \
    df['kw13'], \
    df['kw14'], \
    df['kw15'], \
    df['kw16'], \
    df['kw17'], \
    df['kw18'], \
    df['kw19'], \
    df['kw20'] = df['Keyword'].str.split(' ', 19).str
    # print(df)
    df = df.drop(['Keyword'], axis=1)
    df = df.dropna(subset=['kw1', 'kw2', 'kw3', 'kw4', 'kw5', 'kw6', 'kw7', 'kw8', 'kw9', 'kw10',
                           'kw11', 'kw12', 'kw13', 'kw14', 'kw15', 'kw16', 'kw17', 'kw18', 'kw19', 'kw20'])
    return df


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def text_encode(text_df):
    encode_column = list(text_df.select_dtypes(include=['category', 'object']))
    # print(encode_column)
    le = LabelEncoder()
    for text in encode_column:
        text_df[text] = le.fit_transform(text_df[text].astype(str))
    # print(text_df)
    return text_df


def split_x_y(df):
    y_df = pd.DataFrame()
    y_df['Score'] = df['Score']
    df = df.drop(['Score'], axis=1)
    return df, y_df


def split_x_y_test(test_df):
    test_y_df = pd.DataFrame()
    test_y_df['Score'] = test_df['Score_y']
    test_df = test_df.drop(['Score_y'], axis=1)
    return test_df, test_y_df


def PCA_kw(x_df):
    data = StandardScaler().fit(x_df).transform(x_df)
    pca = PCA(n_components=4).fit_transform(data)
    x_reduct = pd.DataFrame(pca)
    return x_reduct


def gb_classifier(train_df, test_df):
    print(test_df)
    train_df, y_df = split_x_y(train_df)
    test_df, test_y_df = split_x_y_test(test_df)
    test_id_df = pd.DataFrame()
    test_id_df['Id'] = test_df['Id']
    test_df = test_df.drop(['Id'], axis=1)
    #     print(test_id_df)
    #     print(test_df)
    #     print(test_y_df)
    train_df = PCA_kw(train_df)
    test_df = PCA_kw(test_df)

    clf = GradientBoostingClassifier(
        max_depth=15, n_estimators=6000,
        subsample=0.8,
        learning_rate=0.06, random_state=20
    )
    data = clf.fit(train_df, y_df.values.ravel())
    #     print(data)
    pre = clf.predict(test_df)

    result = np.where(pre == -1, 0, pre)
    result = result.tolist()
    final_df = pd.DataFrame()
    final_df['Id'] = test_id_df['Id']
    final_df['Score'] = result
    final_df.to_csv(r'/kaggle/working/logreg_2.csv', index=False)

    print(final_df)


#     print(pre)
# print(pre.score())


imported_df_train, improted_df_test = import_dataX('/kaggle/input/bu-cs-506-fall-2019-midterm-competition/train.csv',
                                                   '/kaggle/input/bu-cs-506-fall-2019-midterm-competition/test.csv')

text_train = extract_keywords(imported_df_train)
text_test = extract_keywords(improted_df_test)
encoded_df_train = text_encode(text_train)
encoded_df_test = text_encode(text_test)
a = gb_classifier(encoded_df_train, encoded_df_test)
