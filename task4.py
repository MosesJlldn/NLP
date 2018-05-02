import string

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV

import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.pipeline import Pipeline


def text_process(text):
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)

    return [word for word in no_punc.split() if word.lower() not in stopwords.words('russian')]


if __name__ == '__main__':
    yelp = pd.read_json('reviews.txt', lines=True)
    # yelp['text length'] = yelp['text'].apply(len)
    yelp = yelp[:100]
    X = yelp['text']
    y = yelp['positive']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
    X_train_counts = bow_transformer.transform(X_train)

    # yelp['text length'] = yelp['text'].apply(len)
    # print(yelp.head())
    # print(yelp.shape)
    # g = sns.FacetGrid(data=yelp, col='positive')
    # g.map(plt.hist, 'text length', bins=50)
    # plt.show()
    # sns.boxplot(x='positive', y='text length', data=yelp)
    # plt.show()

    nb = MultinomialNB()
    nb.fit(X_train_counts, y_train)
    preds = nb.predict(bow_transformer.transform(X_test))
    f1 = open('bayes.txt', 'w+')
    print(confusion_matrix(y_test, preds), file=f1)
    print('\n', file=f1)
    print(classification_report(y_test, preds), file=f1)
    f1.close()

    parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}

    text_clf = Pipeline([('vect', CountVectorizer(analyzer=text_process)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())])

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(X_train, y_train)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
