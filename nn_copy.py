import pandas as pd
import numpy as np
import nltk
import string
import numpy as np
import scipy.sparse as sparse

from collections import Counter
import itertools
import time

from sklearn.cross_validation import train_test_split
from sklearn import metrics
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, recall_score, precision_score

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import *
from keras.callbacks import EarlyStopping


data = pd.read_csv("../data/tokenize_beauty.csv")

max_vocab = 20000
max_features = max_vocab + 1
maxlen = 250
batch_size = 128

def tokenize(data):
	def remove_punc(s):
	    return s.translate(None, string.punctuation)

	def low(s):
	    return s.lower()

	def lem(tokens):
	    lemmatizer = nltk.WordNetLemmatizer()
	    lem_tokens = []
	    for token in tokens:
	        lem_tokens.append(lemmatizer.lemmatize(token))
	    return lem_tokens

	data = data.dropna(subset=["reviewText", "overall"])
	data["sentiment"] = data["overall"].apply(lambda x : 1 if x > 3 else 0)
	data["token"] = data["reviewText"].apply(low).apply(remove_punc).apply(word_tokenize)
	data["token"] = data["token"].apply(lem)

	return data[['sentiment', 'token']]



def prepare_nlp_features(df):
	sentences = []
	for t in df["token"]:
		sentences.append(t)
	word_counts = Counter(itertools.chain(*sentences))
	voc_common = [x[0] for x in word_counts.most_common(max_vocab)]
	voc = {x : i+1 for i, x in enumerate(voc_common)}
	
	X = np.array([[voc.get(word, 0) for word in sentence] for sentence in sentences])
	X = sequence.pad_sequences(X, maxlen = maxlen)
	Y = df['sentiment']

	return X, Y


def NLP_LSTM(data, nb_epoch):

	X, Y = prepare_nlp_features(data)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
	print('Build model...')
	model = Sequential()
	model.add(Embedding(max_features, 128, input_length = maxlen, dropout = 0.3))
	model.add(LSTM(128, dropout_W = 0.3, dropout_U = 0.3))
	model.add(Dense(32))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	early_stopping = EarlyStopping(monitor = 'val_loss', patience = 1)

	model.compile(loss='binary_crossentropy',
            	  optimizer='nadam',
                  metrics=['accuracy'])
	print('Train...')
	model.fit(X_train, y_train, batch_size = batch_size, nb_epoch = nb_epoch, validation_data = (X_test, y_test),
	          callbacks=[early_stopping], verbose = 1)

	y_test_pred = model.predict(X_test, verbose = 0)

	score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
	print('Test score:', score)
	print('Test accuracy:', acc)

	acc = accuracy_score(y_test, (y_test_pred > 0.5).astype(int))
	recall = recall_score(y_test, (y_test_pred > 0.5).astype(int))
	print 'Accuracy: %.2f, Recall %.2f' % (acc, recall)


def NLP_LSTM_Attention(data, nb_epoch):

	X, Y = prepare_nlp_features(data)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
	print('Build model...')
	input = Input(shape = [maxlen])
	embedded = Embedding(max_features, 128, input_length = maxlen, dropout = 0.3, mask_zero = False)(input)
	lstm = LSTM(128, dropout_W = 0.3, dropout_U = 0.3, return_sequences = True)(embedded)

	mask = TimeDistributed(Dense(128, activation = 'softmax'))(lstm)

	merged = merge([lstm, mask], mode = 'mul')
	merged = Flatten()(merged)
	output = Dense(1024, activation = 'relu')(merged)
	output = Dropout(0.2)(output)
	output = Dense(256, activation = 'relu')(output)
	output = Dropout(0.2)(output)
	output = Dense(1, activation = 'linear')(output)
	model = Model(input = input, output = output)

	early_stopping = EarlyStopping(monitor = 'val_loss', patience = 0)

	model.compile(loss='mean_squared_error', optimizer = 'adam')
	print('Train...')
	start = time.time()

	model.fit(X_train, y_train, batch_size = batch_size, nb_epoch = nb_epoch, validation_data = (X_test, y_test),
	          verbose = 1)

	y_test_pred = model.predict(X_test, verbose = 0)
	y_test_pred.shape = (X_test.shape[0], )
	end = time.time()

	acc = accuracy_score(y_test, (y_test_pred > 0.5).astype(int))
	recall = recall_score(y_test, (y_test_pred > 0.5).astype(int))
	precision = precision_score(y_test, (y_test_pred > 0.5).astype(int))
	print 'Accuracy: %.2f, Recall: %.2f, Precision: %.2f' % (acc, recall, precision)

	return acc, recall, precision, end - start, model

# data = tokenize(data)
NLP_LSTM_Attention(data, 20)

# LSTM:
# time = 13 hr
# accuracy = 0.91

# LSTM_attention_hou:
# time = 14 hr
# accuracy = 0.87
# recall = 0.95
# precision = 0.89

# LSTM_attention_wuyi:
# time: 14 hr
# accuracy: 0.87
# recall: 0.95
# precision = 0.88