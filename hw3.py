import nltk
import csv
import math
import numpy as np
import enchant
import string
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, state_union
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk import word_tokenize

def strip_links(text):
    text = re.sub(r"http\S+", "", text)
   
    return text

def strip_all_entities(text): 
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

documents = []

csv_file_name = 'negative.csv'
delimiter_csv = ';'
di = enchant.Dict('en_US')

with open(csv_file_name, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader((line.replace('\0','') for line in csvfile), delimiter = delimiter_csv, quotechar='"')
    for row in reader:
        documents.append(row[3]) 

documents = documents[:100]
filtered_text = []

stop_words = stopwords.words("russian")
stop_words.extend(['rt'])

ps = PorterStemmer()
wnl = WordNetLemmatizer()

for d in documents:
    
    d = strip_all_entities(strip_links(d.lower()))
    tokens = nltk.wordpunct_tokenize(d)    
    filtered_tokens = [w for w in tokens if (w not in stop_words and not di.check(w))]
    stemm_tokens = [(ps.stem(w)) for w in filtered_tokens]
    filtered_text.append(stemm_tokens)

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

words_count_length = len(words_count)

bigram = []
bigram_percent = []

unigram_precent = []
        
for y in range(words_count_length):
    bigram.append([])
    if (y == 0):
        bigram_percent.append(words)
    else:
        bigram_percent.append([])
    for x in range(words_count_length):
        bigram[y].append(0)
        if (y != 0 and x != 0):
            bigram_percent[y].append("0")
        if (x == 0 and y != 0):
            bigram_percent[y].append(words[y])

unigram_precent.append(words)
unigram_precent.append([])

for index, word in enumerate(words):
    unigram_precent[1].append(str(words_count[index] / (words_count_length - 1)))

for tokens in filtered_text:
    token_length = len(tokens)
    for index, token in enumerate(tokens):
        current_index = index
        if (current_index != token_length - 1):
            bigram[words.index(tokens[current_index])][words.index(tokens[current_index + 1])] += 1

for y in range(words_count_length):
    for x in range(words_count_length):
        if (y != 0 and x != 0):
            bigram_percent[y][x] = str(bigram[y][x] / words_count[y])

with open('output_bigram1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for y in range(words_count_length):
        writer.writerow(bigram_percent[y])

with open('output_unigram1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for y in range(2):
        writer.writerow(unigram_precent[y])        
            
print(filtered_text)
