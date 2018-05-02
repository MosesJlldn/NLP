import pandas as pd
import string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk import pos_tag
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

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
    doc = pd.read_csv('stackoverflow_sample_125k.tsv', delimiter='\t', header=-1)
    doc = doc[:100]
    X = doc[0]
    y = doc[1]
    for index, tags in enumerate(y):
        y[index] = tags.split()[0]
    print (X)
    print (y)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    bow_transformer = CountVectorizer(analyzer=text_process).fit(X)
    X_train_counts = bow_transformer.transform(X_train)

    log_reg = LogisticRegression()
    log_reg.fit(X_train_counts, y_train)

    preds = log_reg.predict(bow_transformer.transform(X_test))
    f1 = open('log_reg.txt', 'w+')
    print(confusion_matrix(y_test, preds), file=f1)
    print('\n', file=f1)
    print(classification_report(y_test, preds), file=f1)
    f1.close()
    print ("finish")
