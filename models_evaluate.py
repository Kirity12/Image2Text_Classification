from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, MaxPooling1D, Embedding, LSTM, Bidirectional
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def evaluate(test, pred, description='------------------------------------'):

    print('\n'+ description + ' \n')

    print(classification_report(test,pred)) 
    print('Confusion Matrix:')
    
    print(pd.DataFrame(confusion_matrix(test,pred), index=['0','2','4','6','9'], columns=['0','2','4','6','9']))


def run_NB_classifier(X_train, y_train, X_test, y_test, dat_type):

    # X_train -> X_train_ocr_vectors_tfidf
    # y_train -> train_ocr_dataset['class']
    # X_test -> X_val_ocr_vectors_tfidf
    
    filename = 'models/'+dat_type+'/naiveBayes.sav'

    try:
        nb = pickle.load(open(filename, 'rb'))
    except:
        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        pickle.dump(nb, open(filename, 'wb'))
        

    y_predict = nb.predict(X_test)

    evaluate(y_test, y_predict)


def run_LogisticReg_classifier(X_train, y_train, X_test, y_test, dat_type):

    # X_train -> X_train_ocr_vectors_tfidf
    # y_train -> train_ocr_dataset['class']
    # X_test -> X_val_ocr_vectors_tfidf

    filename = 'models/'+dat_type+'/logisticRegression.sav'

    try:
        lr = pickle.load(open(filename, 'rb'))
    except:
        lr = LogisticRegression(multi_class='multinomial', penalty='l2')
        lr.fit(X_train, y_train)  
        pickle.dump(lr, open(filename, 'wb'))
    
    y_predict = lr.predict(X_test)

    evaluate(y_test, y_predict)


def run_SVM_classifier(X_train, y_train, X_test, y_test, dat_type):

    # X_train -> X_train_ocr_vectors_tfidf
    # y_train -> train_ocr_dataset['class']
    # X_test -> X_val_ocr_vectors_tfidf

    filename = 'models/'+dat_type+'/supportVector.sav'

    try:
        svc = pickle.load(open(filename, 'rb'))
    except:
        svc = svm.SVC(decision_function_shape='ovr')
        svc.fit(X_train, y_train)  
        pickle.dump(svc, open(filename, 'wb'))

    y_predict = svc.predict(X_test)

    evaluate(y_test, y_predict)


def run_cnn_architecture(X_train, y_train, X_test, y_test, dat_type):

    c2i_dir = {'0':0,'2':1,'4':2,'6':3,'9':4}
    i2c_dir = {0:'0',1:"2",2:'4',3:'4',4:'9'}

    # X_train -> X_train_txt_vectors_w2v
    # y_train -> train_txt_dataset['class']
    # X_test -> X_test_txt_vectors_w2v
    # y_test -> test_txt_dataset['class']

    filename = 'models/'+dat_type+'/cnn_model'

    try:
        model = load_model(filename)
    except:

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        y_train = y_train.map(c2i_dir)

        y_train = to_categorical(y_train, num_classes=5)

        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1],1)))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(5, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(X_train,y_train, epochs=30, batch_size=32)
        model.save(filename)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    y_pred = model.predict(X_test)

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_pred_classes = [i2c_dir[x] for x in y_pred_classes]

    evaluate(y_test, y_pred_classes)


def run_bidirectional_LSTM_architecture(X_train, y_train, X_test, y_test, embedding_matrix, vocab_size, dat_type):

    embedding_dim = 128
    num_filters = 128
    filter_sizes = [3, 4, 5]
    dropout_rate = 0.5
    lstm_units = 64

    c2i_dir = {'0':0,'2':1,'4':2,'6':3,'9':4}
    i2c_dir = {0:'0',1:"2",2:'4',3:'4',4:'9'}

    filename = 'models/'+dat_type+'/bidirectional_lstm'
    try:
        model = load_model(filename)
    except:
        y_train = y_train.map(c2i_dir)

        y_train = to_categorical(y_train, num_classes=5)

        embedding_dim = 128
        num_filters = 128
        filter_sizes = [3, 4, 5]
        dropout_rate = 0.5
        lstm_units = 64

        # max_features = 2000
        # tokenizer = Tokenizer(num_words = max_features, )
        # ds = pd.concat([train_txt_dataset, test_txt_dataset, val_txt_dataset])
        # tokenizer.fit_on_texts(ds['clean_text'].values)

        # X_train = tokenizer.texts_to_sequences(train_txt_dataset['clean_text'].values)
        # X_train = pad_sequences(X_train, padding = 'post' ,maxlen=128)

        # vocab_size = len(tokenizer.word_index)+1
        # vocab_size, X_train.shape

        model = Sequential()

        model.add(Embedding(vocab_size, embedding_dim, input_length =X_train.shape[1], weights = [ embedding_matrix] , trainable = False))

        conv_blocks = []
        for filter_size in filter_sizes:
            conv_layer = Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(model.layers[-1].output)
            maxpool_layer = MaxPooling1D(pool_size=128 - filter_size + 1)(conv_layer)
            conv_blocks.append(maxpool_layer)

        model.add(Dropout(dropout_rate))
        model.add(Dense(units=lstm_units, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=True)))
        model.add(Bidirectional(LSTM(units=lstm_units)))
        model.add(Dense(units=5, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


        model.fit(X_train,y_train, epochs=30, batch_size=32)

        model.save(filename)

    y_pred = model.predict(X_test)

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_pred_classes = [i2c_dir[x] for x in y_pred_classes]

    evaluate(y_test, y_pred_classes)

