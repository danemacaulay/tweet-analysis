import pandas as pd
from collections import Counter
from nltk import ngrams
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def strip_links(x):
    stripped = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", x)
    return stripped

def remove_stopwords(x):
    word_tokens = word_tokenize(x)
    filtered = [w for w in word_tokens if w.lower() not in stop_words if w.isalpha()]
    return " ".join(filtered)

df = pd.read_json('data/all.json')
df["likes_normalized"] = df["likes"] / df["likes"].max()
df["replies_normalized"] = df["replies"] / df["replies"].max()
df["retweets_normalized"] = df["retweets"] / df["retweets"].max()
df = df[(df["likes_normalized"] > 0.05) | (df["replies_normalized"] > 0.05) | (df["retweets_normalized"] > 0.05)]
# df = df[(df["likes_normalized"] < 0.0001) | (df["replies_normalized"] < 0.0001) | (df["retweets_normalized"] < 0.0001)]
df['text'] = df['text'].apply(lambda x: x.replace('http', ' http'))
df['text'] = df['text'].apply(strip_links)
df['text'] = df['text'].apply(remove_stopwords)
print(len(df))
texts = df['text'].tolist()

ngram_counts = Counter(ngrams(" ".join(texts).split(), 2))
counts = ngram_counts.most_common(20)
print(counts)


