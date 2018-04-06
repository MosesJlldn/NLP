import nltk
import csv
import math
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, state_union
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from collections import Counter

def token_tf(word, tokens_all):
    count = 0
    for tokens in tokens_all:
        count += tokens.count(word)
    
    return count

def token_df(word, tokens_all):
    count = 0
    for tokens in tokens_all:
        if word in tokens:
            count += 1
    
    return count

##example_text = ""
documents = []

with open('literature_2010.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        documents.append(row[3])

documents = documents[:2]
filtered_text = []

stop_words = set(stopwords.words("english"))
tokenizer = RegexpTokenizer(r'\w+')

ps = PorterStemmer()
wnl = WordNetLemmatizer()

for d in documents:
    
    d = d.lower()
    tokens = tokenizer.tokenize(d)
    filtered_tokens = [w for w in tokens if not w in stop_words]
    stemm_tokens = [(ps.stem(w)) for w in filtered_tokens]
    filtered_text.append(stemm_tokens)

di = {}

for tokens in filtered_text:
    for token in tokens:
        if token not in di:
            di[token] = token_tf(token, filtered_text) * math.log2(len(filtered_text) / token_df(token, filtered_text))

di = Counter(di)
##for k, v in di.most_common(5):
##   print(k, v)

print(filtered_text)

words = []
words.append("")
words_count = []
words_count.append(0)


for tokens in filtered_text:
    for token in tokens:
        if token not in words:
            words.append(token)
            words_count.append(1)
        else:
            words_count[words.index(token)] += 1

print(words_count)
words_count_length = len(words_count)
#print(words)
#print(words_count)

bigram = []
bigram_percent = []
        
for y in range(words_count_length):
    bigram.append([])
    bigram_percent.append([])
    for x in range(words_count_length):
        bigram[y].append(0)
        bigram_percent[y].append(0)

for tokens in filtered_text:
    token_length = len(tokens)
    for token in tokens:
        current_index = tokens.index(token)
        if (current_index != token_length - 1):
            bigram[words.index(tokens[current_index])][words.index(tokens[current_index + 1])] += 1

for y in range(words_count_length):
    for x in range(words_count_length):
        if (y != 0 and x != 0):
            bigram_percent[y][x] = bigram[y][x] / words_count[y]


print(words)
for y in range(len(words_count)):
    print(bigram_percent[y])
for y in range(len(words_count)):
    print(bigram[y])    
