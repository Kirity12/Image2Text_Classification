from gensim.models import Word2Vec
import nltk
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


class TextEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(next(iter(word2vec.values())))
            
    def transform(self, X):
            return np.array([
                np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                        or [np.zeros(self.dim)], axis=0)
                for words in X
            ])


def initiate_tokenizer(dataset, max_features=2000):

    # dataset -> alldata['clean_text'].values

    tokenizer = Tokenizer(num_words = max_features, )
    tokenizer.fit_on_texts(dataset)
    return tokenizer


def tokenize_data(tokenizer, data, seq_len=128):
    # data->train_txt_dataset['clean_text'].values

    x = tokenizer.texts_to_sequences(data)
    x = pad_sequences(x, padding = 'post' ,maxlen=seq_len)
    return x


def initiate_tfidf_vectorizer(data):

    # data -> train['clean_text']
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_vectorizer.fit(data)

    return tfidf_vectorizer


def generate_tfidf_vector(tfidf_vectorizer, data):

    # data -> train['clean_text']

    X_test_tfidf = tfidf_vectorizer.transform(data)

    return X_test_tfidf


def initiate_w2v_model(tokens, dat_typ='txt', vectorsize=128):

    model = Word2Vec(tokens, vector_size=vectorsize, window=5, min_count=1, workers=4)
    model.train(tokens,epochs=30,total_examples=len(tokens))

    model.wv.save_word2vec_format('word_embeddings_'+dat_typ+'.txt', binary=False)
    return model


def generate_w2v_word_embedding_vectors(tokens, tokenizer, dat_typ='txt', vectorsize=128):

    if not os.path.exists('word_embeddings_'+dat_typ+'.txt'):
        initiate_w2v_model(tokens, dat_typ, vectorsize)

    embedding_vector = {}
    with open('word_embeddings_'+dat_typ+'.txt') as f:
        for line in tqdm(f):
            value = line.split(' ')
            word = value[0]
            coef = np.array(value[1:],dtype = 'float32')
            embedding_vector[word] = coef

    embedding_matrix = np.zeros((len(tokenizer.word_index)+1, vectorsize))
    for word,i in tqdm(tokenizer.word_index.items()):
        embedding_value = embedding_vector.get(word)
        if embedding_value is not None:
            embedding_matrix[i] = embedding_value 

    return embedding_matrix


def nltk_tokenize(data):
    return [nltk.word_tokenize(i) for i in data]


def generate_text_w2v_embeddings(model, data):

    # data->train_dataset['clean_text']

    w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))
    modelw = TextEmbeddingVectorizer(w2v)

    tok = [nltk.word_tokenize(i) for i in data]

    vec = modelw.transform(tok)
    
    return vec


if __name__=="__main__":

    text = "In linguistic morphology, inflection is a process of word formation, in which a word is modified to express different grammatical categories such as tense, case, voice, aspect, person, number, gender, mood, animacy, and definiteness."
