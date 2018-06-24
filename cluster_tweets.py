import pandas as pd
import spacy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import decomposition
from joblib import Memory

memory = Memory(cachedir='cache', verbose=0)


def preprocess_df(df):
    vectors = calculate_vectors(df, 'en_core_web_lg')
    # vectors = calculate_vectors(df, 'models/twitter-27B-200')
    df['vectors'] = list(np.array(vectors))
    df["likes_normalized"] = df["likes"] / df["likes"].max()
    df["replies_normalized"] = df["replies"] / df["replies"].max()
    df["retweets_normalized"] = df["retweets"] / df["retweets"].max()
    df['text'] = df['text'].apply(lambda x: x.replace('http', ' http'))
    df = df[(df["likes_normalized"] > 0.05) | (df["replies_normalized"] > 0.05) | (df["retweets_normalized"] > 0.05)]
    return df

@memory.cache
def calculate_vectors(df, model):
    nlp = spacy.load(model)
    texts = df['text'].tolist()
    vectors = []
    for t in texts:
        doc = nlp(t)
        vectors.append(doc.vector)
    return vectors


def build_features(df):
    features_list = []
    for index, row in df.iterrows():
        features_list.append(row['vectors'])
    return features_list


def run_silouhette_analysis(features_list):
    range_n_clusters = list(range(2, 30))
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(features_list)
        silhouette_avg = silhouette_score(features_list, clusters)
        print('number of clusters: {}, silhoutte avg: {}'.format(n_clusters, silhouette_avg))

def build_clusters(df, features_list):
    # pca = decomposition.PCA(n_components=20)
    # features_list = pca.fit_transform(features_list)
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(features_list)
    df['clusters'] = clusters
    df = df.drop(['html','vectors', 'likes_normalized', 'replies_normalized', 'retweets_normalized', 'user'], axis=1)
    df = df.sort_values(by=['clusters'])
    df.to_csv('data/clustering_analysis.csv')

df = pd.read_json('data/all.json')
preprocessed_df = preprocess_df(df)
features_list = build_features(preprocessed_df)
run_silouhette_analysis(features_list)
# build_clusters(df, features_list)
