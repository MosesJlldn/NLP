import pandas as pd
import string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def text_process(text):
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)

    return [word for word in no_punc.split() if word.lower() not in stopwords.words('russian')]


if __name__ == '__main__':
    doc = pd.read_csv('stackoverflow_sample_125k.tsv', delimiter='\t', header=-1)
    doc = doc[:10]
    X = doc[0]
    y = doc[1]
    for index, tags in enumerate(y):
        y[index] = tags.split()[0]
    print (X)
    print (y)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
    X_train_counts = bow_transformer.transform(X_train)

    parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
    text_clf = Pipeline([('vect', CountVectorizer(analyzer=text_process)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LogisticRegression())])

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(X_train, y_train)

    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
