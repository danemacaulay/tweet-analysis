import json
import scipy
from pandas.io.json import json_normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import gensim
from annoy import AnnoyIndex

def load_vectors(filename, binary):
    vectors_model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=binary)
    vectors_vocab = [" ".join(k.split('_')) for k in vectors_model.vocab.keys()]
    print('Vectors Loaded: ', len(vectors_vocab))
    return vectors_model, vectors_vocab


def load_fast_text_vectors(filename):
    vectors_model = FastText.load_fasttext_format(filename)
    vectors_vocab = [" ".join(k.split('_')) for k in vectors_model.vocab.keys()]
    return vectors_model, vectors_vocab


def normalize_data(filename):
    df = pd.read_csv(filename, dtype={'duns':str})
    # df.drop(df[df['text'].map(len) < 50].index, inplace=True)
    # df = df[df['top-terms'].str.contains("##") == False]
    # df['text'].replace('', np.nan, inplace=True)
    # df.dropna(subset=['text'], inplace=True)
    print('Companies loaded: ', len(df))
    return df


def init_tfidf(vectors_vocab):
    return TfidfVectorizer(stop_words='english',
                            min_df=2,
                            vocabulary=vectors_vocab,
                            ngram_range=(1, 4))


def compute_vectors(tfidf, df, vectors_model):
    print('Fitting TFIDF')
    tfidf_vector_list = tfidf.fit_transform(df['text'])
    term_to_index = tfidf.vocabulary_
    index_to_term = {v: k for k, v in term_to_index.items()}
    cx = scipy.sparse.coo_matrix(tfidf_vector_list)
    print('Computing vectors')
    computed_vector_map = {}
    for row, index, freq in zip(cx.row, cx.col, cx.data):
        term = index_to_term[index]
        term_with_underscore = "_".join(term.split(" "))
        vector = vectors_model[term_with_underscore]
        if row in computed_vector_map:
            computed_vector_map[row] += vector * freq
        else:
            computed_vector_map[row] = vector * freq
            if row % 10000 == 0:
                print('row:', row, '/', len(df))
    return computed_vector_map


def build_index(computed_vector_map, vector_length, n_trees):
    print('Building index')
    annoy_index = AnnoyIndex(vector_length, metric='angular') # "euclidean", "manhattan", or "hamming"
    for key, value in computed_vector_map.items():
        annoy_index.add_item(key, value)
    annoy_index.build(n_trees)
    return annoy_index


if __name__ == '__main__':
    vectors_model, vectors_vocab = load_vectors('./embeddings/glove.twitter.27B.200d.w2v.txt', False)
    df = normalize_data('list.csv')
    tfidf = init_tfidf(vectors_vocab)
    computed_vectors = compute_vectors(tfidf, df, vectors_model)
    index = build_index(computed_vectors, vector_length=50, n_trees=50)

    print(index.get_nns_by_item(1, 10, 10))
    print(index.get_nns_by_vector(vectors_model['cooling'], 10, 10))
