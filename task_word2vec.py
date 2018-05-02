import gensim, logging
import nltk
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from nltk import pos_tag


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin


class AvgFeatureVec(BaseEstimator, TransformerMixin):

    def __init__(self, model, num_features=150):
        self.model = model
        self.num_features = num_features

    def get_feature_names(self):
        feature_names = []
        for i in range(150):
            feature_names.append('feature_%s' %i)
        return feature_names

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            features.append(
                self.makeFeatureVec(nltk.word_tokenize(text),
                                    self.model,
                                    self.num_features))
        return features

    def makeFeatureVec(self, words, model, num_features):
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0.
        index2word_set = set(model.wv.vocab)
        for word in words:
            if word in index2word_set:
                nwords += 1.
                featureVec = np.add(featureVec, model[word])

        featureVec = np.divide(featureVec, nwords)
        return featureVec


def text_process(text):
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)

    words = []
    for word in no_punc.split():
        if word.lower() not in stopwords.words('english'):
            words.append(word)
            words.append(pos_tag([word], tagset="universal")[0][1])

    return words


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print('Reading...')
    yelp = pd.read_csv('stackoverflow_sample_125k.tsv', sep='\t', header=-1)
    yelp = yelp[:100]
    X = yelp[0]
    y_common = yelp[1]
    y = []
    for tags in y_common:
        y.append(tags.split(' ')[0])

    X_tokens = []
    for text in X:
        X_tokens.append(nltk.word_tokenize(text))

    print('Training w2v...')
    model = gensim.models.Word2Vec(X_tokens, workers=4, size=150)
    f1 = open('word2vec.txt', 'w+')
    print(model.most_similar("android"), file=f1)
    print(model.most_similar("java"), file=f1)
    print(model.most_similar("program"), file=f1)
    f1.close()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    clf = Pipeline([('feats', FeatureUnion([
        ('count_vect', CountVectorizer(analyzer=text_process)),
        ('words_avg_vec', AvgFeatureVec(model, 150))
                    ])),
                    ('clf', LogisticRegression(multi_class='ovr'))
                    ])

    print('Training log reg...')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f2 = open('log_reg_with_word2vec.txt', 'w+')
    print(classification_report(y_test, y_pred), file=f2)
    f2.close()
    print('Done Sir')

