from dataset import data_loader
import pandas as pd
from text_preprocess import finalpreprocess
from generate_embeddings import *
from models_evaluate import *
import warnings
warnings.filterwarnings("ignore")


dat_type = 'txt'
complete_dataset, train_dataset, test_dataset, val_dataset = data_loader(dat_type)

train_dataset['clean_text'] = train_dataset['text'].apply(lambda x: finalpreprocess(x))
test_dataset['clean_text'] = test_dataset['text'].apply(lambda x: finalpreprocess(x))
val_dataset['clean_text'] = val_dataset['text'].apply(lambda x: finalpreprocess(x))

print('================================data processed=============================')

provided_set = pd.concat([train_dataset, test_dataset])

tfidf_vectorizer = initiate_tfidf_vectorizer(provided_set['clean_text'])

X_train_vectors_tfidf = generate_tfidf_vector(tfidf_vectorizer,train_dataset['clean_text']) 
X_test_vectors_tfidf = generate_tfidf_vector(tfidf_vectorizer,test_dataset['clean_text'])
X_val_vectors_tfidf = generate_tfidf_vector(tfidf_vectorizer,val_dataset['clean_text'])

tokenizer = initiate_tokenizer(provided_set['clean_text'].values)

X_train_tokens = tokenize_data(tokenizer, train_dataset['clean_text'].values) 
X_test_tokens = tokenize_data(tokenizer, test_dataset['clean_text'].values)
X_val_tokens = tokenize_data(tokenizer, val_dataset['clean_text'].values)

nltk_train_tokens = nltk_tokenize(train_dataset['clean_text'])
nltk_test_tokens = nltk_tokenize(test_dataset['clean_text'])
nltk_val_tokens = nltk_tokenize(val_dataset['clean_text'])

w2v_vectorizer = initiate_w2v_model(nltk_train_tokens+nltk_val_tokens,dat_typ=dat_type)
word_embedding_matrix = generate_w2v_word_embedding_vectors(X_train_tokens, tokenizer, dat_typ=dat_type)

X_train_vectors_w2v = generate_text_w2v_embeddings(w2v_vectorizer,train_dataset['clean_text']) 
X_test_vectors_w2v = generate_text_w2v_embeddings(w2v_vectorizer,test_dataset['clean_text'])
X_val_vectors_w2v = generate_text_w2v_embeddings(w2v_vectorizer,val_dataset['clean_text'])

print('================================Embeddings Generated=============================')


print('\n------------Naive Bayes Classifier---------------')
run_NB_classifier(X_train_vectors_tfidf, train_dataset['class'], X_val_vectors_tfidf, val_dataset['class'],dat_type)
print('\n------------Multinomila Logistic Regression Classifier---------------')
run_LogisticReg_classifier(X_train_vectors_tfidf, train_dataset['class'], X_val_vectors_tfidf, val_dataset['class'],dat_type)
print('\n------------Support Vevtor Classifier---------------')
run_SVM_classifier(X_train_vectors_tfidf, train_dataset['class'], X_val_vectors_tfidf, val_dataset['class'],dat_type)
print('\n------------CNN Classifier---------------')
run_cnn_architecture(X_train_vectors_w2v,train_dataset['class'],X_val_vectors_w2v,val_dataset['class'],dat_type)
print('\n------------Bidirectional LSTM Classifier---------------')
run_bidirectional_LSTM_architecture(X_train_tokens,train_dataset['class'],X_val_tokens,val_dataset['class'], word_embedding_matrix, len(tokenizer.word_index)+1,dat_type)

print('================================Models Evaluated=============================')
