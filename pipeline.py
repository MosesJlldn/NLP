import nltk, csv, string, re, enchant, math
from nltk.stem.snowball import RussianStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

filename = 'positive.csv'
d = enchant.Dict('en_US')
stem = RussianStemmer()

def strip_links(text):
    #link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    #links         = re.findall(link_regex, text)
    #for link in links:
    #    text = text.replace(link[0], '') 
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

texts = []
##with open(filename, "r", newline="") as file:
##    reader = csv.reader((line.replace('\0','') for line in file), delimiter=';')
##    for row in reader:
##        texts.append(row[3])

csv_file_name = 'positive.csv'
delimiter_csv = ';'
##csv_file_name = 'literature_2010.csv'

with open(csv_file_name, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader((line.replace('\0','') for line in csvfile), delimiter = delimiter_csv, quotechar='"')
    for row in reader:
        texts.append(row[3])

texts = [strip_all_entities(strip_links(text.lower())) for text in texts]
texts = texts[:3]

tokens_all = [nltk.wordpunct_tokenize(text) for text in texts]
#for tokens in tokens_all:
#    tokens = [i for i in tokens if (i not in string.punctuation)]
tokens_all = [[i for i in tokens if (i not in string.punctuation)] for tokens in tokens_all]

stop = stopwords.words('russian')
stop.extend(['rt'])
#for tokens in tokens_all:
#    tokens = [stem.stem(i) for i in tokens if (i not in stop and not d.check(i))]

tokens_all = [[stem.stem(i) for i in tokens if (i not in stop and not d.check(i))]for tokens in tokens_all]

print(tokens_all)

di = {}
for tokens in tokens_all:
    for token in tokens:
        if token not in di:
            di[token] = token_tf(token, tokens_all) * math.log2(len(tokens_all) / token_df(token, tokens_all))
            
di = Counter(di)
for k, v in di.most_common(5):
    print(k, v)
