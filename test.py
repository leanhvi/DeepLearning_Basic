
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np


dict_path =  "/home/levi/vi/challenges/cmt-data/dicts/id_full.txt"
stopwords_path = "/home/levi/vi/challenges/cmt-data/dicts/stop_words.txt"

normal_train_path = "/home/levi/vi/challenges/cmt-data/training_data/normal_comments.txt"
sara_train_path = "/home/levi/vi/challenges/cmt-data/training_data/sara_comments.txt"

normal_test_path = "/home/levi/vi/challenges/cmt-data/test_data/nornal_comments.txt"
sara_test_path = "/home/levi/vi/challenges/cmt-data/test_data/sara_comments.txt"

min_freq = 5
start_index = 3

min_comment_length = 3
max_comment_length = 30

vocab = {}
stopwords = set()

def get_stopwords():
    global stopwords
    f = open(stopwords_path, 'r')    
    for line in f:
        stopwords.add(line.strip())    
    f.close()    
    return

def get_vocab():
    global vocab
    f = open(dict_path, 'r')
    index = start_index
    for line in f:
        parts = line.split()
        token = parts[0]
        freq = int(parts[1])
        if freq < min_freq:
            break
        if token in stopwords:
#            print("Encounter a stopword %s:" % token)
            continue
        vocab[token] = index
        index += 1        
    f.close()
    return    

def get_data(path):
    f = open(path, 'r')    
    data = []
    for line in f:
        tokens = line.split()        
        encoded = []        
        for token in tokens:
            if token in vocab:
                code = vocab[token]
                encoded.append(code)
        if len(encoded) < min_comment_length:
            continue            
        if len(encoded) >= max_comment_length:
            encoded = np.array(encoded, dtype = np.int)
            encoded = encoded[0:max_comment_length]   
        else:
            padding = np.zeros((max_comment_length - len(encoded)), dtype = np.int)
            encoded = np.concatenate((padding, encoded))                    
            
        data.append(encoded)
    
    f.close
    return np.array(data)
    

get_stopwords()
get_vocab()

normal_comment_train = get_data(normal_train_path)
normal_labels_train = np.ones((len(normal_comment_train)), dtype = np.int)
sara_comment_train = get_data(sara_train_path)
sara_labels_train = np.zeros(len(sara_comment_train), dtype = np.int)

X_train = np.concatenate((normal_comment_train, sara_comment_train), axis = 0)
y_train = np.concatenate((normal_labels_train, sara_labels_train), axis = 0)


normal_comment_test = get_data(normal_test_path)
normal_labels_test = np.ones((len(normal_comment_test)), dtype = np.int)
sara_comment_test = get_data(sara_test_path)
sara_labels_test = np.zeros(len(sara_comment_test), dtype = np.int)

X_test = np.concatenate((normal_comment_test, sara_comment_test), axis = 0)
y_test = np.concatenate((normal_labels_test, sara_labels_test), axis = 0)




# LSTM for sequence classification in the IMDB dataset

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
#from keras.preprocessing import sequence

# fix random seed for reproducibility
np.random.seed(7)


# create the model
top_words = len(vocab)
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_comment_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, nb_epoch=5, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
