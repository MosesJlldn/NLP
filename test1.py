import nltk
import csv
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, state_union
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

'''
example_text = "Hello Mr. Smith, how are you doing today? The weather is great and Python is awesome. The sky is pinkish_blue. You should not eat cardboard."
print(sent_tokenize(example_text))
print(word_tokenize(example_text))
for i in word_tokenize(example_text):
    print(i)
'''

##ps = PorterStemmer()
##
##example_words = ["python","pythoner","pythoning","pythoned","pythonly"]
##
##for w in example_words:
##    print(ps.stem(w))

##for w in words:
##    if w not in stop_words:
##        filtered_text.append(w)

#words = word_tokenize(example_text)

#example_text = state_union.raw("literature_2010.csv");

##example_text = "This is, an example, showing off stop word filtration."

example_text = ""
documents = []

with open('literature_2018.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        #example_text += ' ' + row[3]
        documents.appennd(row[3])
            
##example_text = field
##example_text = field.lower()

example_text = example_text.lower()

for d in documents:

stop_words = set(stopwords.words("english"))

tokenizer = RegexpTokenizer(r'\w+')


words = tokenizer.tokenize(example_text)
filtered_text = []


filtered_text = [w for w in words if not w in stop_words]
stemm_text = []
ps = PorterStemmer()
lemm_text = []
wnl = WordNetLemmatizer()

for w in filtered_text:
    stemm_text.append(ps.stem(w))

for w in stemm_text:
    lemm_text.append(wnl.lemmatize(w))

print(lemm_text)


