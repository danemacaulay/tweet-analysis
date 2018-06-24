import pandas as pd
import spacy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import decomposition
from joblib import Memory
import numpy as np
import pandas
from nltk import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
import semantic_scoring as ss
import re
memory = Memory(cachedir='cache', verbose=0)

def strip_links(x):
    stripped = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", x)
    return stripped

def preprocess_df(df):
    df['text'] = df['text'].apply(lambda x: x.replace('http', ' http'))
    df['text'] = df['text'].apply(strip_links)
    df["likes_normalized"] = df["likes"] / df["likes"].max()
    df["replies_normalized"] = df["replies"] / df["replies"].max()
    df["retweets_normalized"] = df["retweets"] / df["retweets"].max()
    df['text'] = df['text'].apply(lambda x: x.replace('http', ' http'))
    # df = df[(df["likes_normalized"] < 0.0001) | (df["replies_normalized"] < 0.0001) | (df["retweets_normalized"] < 0.0001)]
    df = df[(df["likes_normalized"] > 0.05) | (df["replies_normalized"] > 0.05) | (df["retweets_normalized"] > 0.05)]
    print(len(df))
    return df

def get_topics(df_series, no_topics=20, no_top_words=10):
    tfidf_vectorizer = TfidfVectorizer(min_df=2, stop_words='english', ngram_range=[1,3])
    tfidf = tfidf_vectorizer.fit_transform(df_series)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    # tf_vectorizer = CountVectorizer(min_df=2, stop_words='english')
    # tf = tf_vectorizer.fit_transform(df_series)
    # tf_feature_names = tf_vectorizer.get_feature_names()
    nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    # display_topics(nmf, tfidf_feature_names, no_top_words)
    clusters = nmf.transform(tfidf)
    return clusters

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_name = get_topic_label(words)
        print("'{}':".format(topic_name))
        print(", ".join(words))
        print()

def get_topic_label(topics):
    summed_vector = [0] * 300
    for topic in topics:
        if topic in vectors_model_set:
            summed_vector = summed_vector + vectors_model[topic]
        else:
            child_topics = topic.split()
            for ctopic in child_topics:
                if ctopic in vectors_model_set:
                    summed_vector = summed_vector + vectors_model[ctopic]
    topic_name = vectors_model.most_similar(positive=[summed_vector], topn=1)[0][0]
    return topic_name

# VECTORS_FILE = 'embeddings/glove.w2v.6B.300d.txt'
# VECTORS_FILE_BINARY = False
# vectors_model, vectors_vocab = ss.load_vectors(VECTORS_FILE, VECTORS_FILE_BINARY)
# vectors_model_set = set(vectors_vocab)

df = pd.read_json('data/all.json')
df = preprocess_df(df)
clusters = get_topics(df['text'], no_topics=10, no_top_words=20)
df["clusters"] = ""
df["clusters"] = clusters
df = df.drop(['html', 'likes_normalized', 'replies_normalized', 'retweets_normalized', 'user'], axis=1)
df = df.sort_values(by=['clusters'])
df.to_csv('output/clustering_analysis-nmf-10.csv')

