import string

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk import pos_tag
import numpy as np
import matplotlib.pyplot as plt


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
    yelp = pd.read_csv('stackoverflow_sample_125k.tsv', sep='\t', header=-1)
    yelp = yelp[:5000]
    X = yelp[0]
    y_common = yelp[1]
    y = []
    for tags in y_common:
        y.append(tags.split(' ')[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
    X_train_counts = bow_transformer.transform(X_train)

    # yelp[2] = yelp[0].apply(len)
    # g = sns.FacetGrid(data=yelp, col=1)
    # g.map(plt.hist, 2, bins=50)
    # plt.show()
    # sns.boxplot(x=1, y=2, data=yelp)
    # plt.show()

    log_reg = LogisticRegression()
    log_reg.fit(X_train_counts, y_train)

    preds = log_reg.predict(bow_transformer.transform(X_test))
    f1 = open('log_reg5000.txt', 'w+')
    print(confusion_matrix(y_test, preds), file=f1)
    print('\n', file=f1)
    print(classification_report(y_test, preds), file=f1)
    f1.close()

    # coefs = log_reg.coef_.tolist()
    # f2 = open('log_reg_most_inform_params.txt', 'w+')
    # tab = "    "
    # for i in range(len(coefs)):
    #     print(log_reg.classes_[i], file=f2)
    #     for j in range(10):
    #         max_index = coefs[i].index(max(coefs[i]))
    #         coefs[i][max_index] = -1
    #         word = list(bow_transformer.vocabulary_.keys())[list(bow_transformer.vocabulary_.values()).index(max_index)]
    #         print(tab, word, file=f2)
    #
    # f2.close()
    print("DONE")
