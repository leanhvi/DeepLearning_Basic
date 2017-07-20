
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np
import codecs

'''CONSTANTS AND UTILITIES FUNCTION'''

dict_path =  "/home/levi/vi/challenges/cmt-data/dicts/id_full.txt"
stopwords_path = "/home/levi/vi/challenges/cmt-data/dicts/stop_words.txt"

normal_train_path = "/home/levi/vi/challenges/cmt-data/training_data/normal_comments.txt"
sara_train_path = "/home/levi/vi/challenges/cmt-data/training_data/sara_comments.txt"

normal_test_path = "/home/levi/vi/challenges/cmt-data/test_data/nornal_comments.txt"
sara_test_path = "/home/levi/vi/challenges/cmt-data/test_data/sara_comments.txt"

model_saving_path = "/home/levi/vi/challenges/cmt-data/sara_model.h5"

min_freq = 5
start_index = 3

min_comment_length = 3
max_comment_length = 50

''' id=2 for words that's not exist in vocabulary'''
out_of_vocab = 2 

''' id=1 for start new sentence character'''
new_sentence = 1  

'''Get collection of stopwords'''
def get_stopwords():    
    stopwords = set()
    f = codecs.open(stopwords_path, 'r', encoding='utf8')    
    for line in f:
        stopwords.add(line.strip())    
    f.close()    
    return stopwords

'''Get dictionary map words to it's id
    { word --> id }
'''
def get_vocab():
    vocab = {}
    vocab['.'] = new_sentence
    f = codecs.open(dict_path, 'r', encoding='utf8')
    index = start_index
    for line in f:
        parts = line.split()
        token = parts[0]
        freq = int(parts[1])
        if freq < min_freq:
            break
        vocab[token] = index
        index += 1        
    f.close()
    return vocab   

'''
Read comment in text file into an 2-D array Nx50: N is number of valid comments  and 50 is Max Comment length
    1st-D -> Comments id; 
    2nd-D -> encoded words (id of words) in a comments
    [
        [ x00, x01, x02, ...],
        [ x10, x11, x12, ...]
        ...
    ]
    padding 0 if the comment length < 50
    truncate if the comment length > 50 
'''
def get_data(vocab, stopwords, path):
    f = codecs.open(path, 'r', encoding='utf8')    
    data = []
    for line in f:
        line = line.lower().replace('.', ' . ')
        tokens = line.split()        
        encoded = [new_sentence]        
        for token in tokens:
            if token in stopwords:
                continue
            
            if token in vocab:
                code = vocab[token]
            else:
                code = out_of_vocab
                
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

'''Transform float value array to binary array by threshold'''
def to_binary(Y, threshold):
    l = len(Y)
    for i in range(l):
        if Y[i] >= threshold:
            Y[i] = 1
        else:
            Y[i] = 0
    return Y



'''LOAD TRANING DATA'''

stopwords = get_stopwords()
vocab = get_vocab()

normal_comment_train = get_data(vocab, stopwords, normal_train_path)
'''Sara comments have labels zero'''
normal_labels_train = np.ones((len(normal_comment_train)), dtype = np.int)

sara_comment_train = get_data(vocab, stopwords, sara_train_path)
'''Sara comments have labels zero'''
sara_labels_train = np.zeros(len(sara_comment_train), dtype = np.int)

'''Concatenate the Normal set with Sara set'''
X_train = np.concatenate((normal_comment_train, sara_comment_train), axis = 0)
y_train = np.concatenate((normal_labels_train, sara_labels_train), axis = 0)




'''TRAINING PHASE '''
'''Using Recurrent Neutral Network - LSTM Unit for sequence classification'''

'''Preprocess data: Shuffle the training set to merge Normal set with Sara set'''
np.random.seed(7)
np.random.shuffle(X_train)
np.random.seed(7)
np.random.shuffle(y_train)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding


'''length of vocabulary plus out_of_vob and padding character'''
total_words = len(vocab) + 2 
embedding_vector_length = 100


def train(X_train, y_train):
    model = Sequential()
    model.add(Embedding(total_words, embedding_vector_length, input_length=max_comment_length))
    model.add(LSTM(32))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=3, batch_size=100)    
    return model

'''Train and save model'''
np.random.seed(7)
model = train(X_train, y_train)
model.save(model_saving_path)
    


'''TESTING PHASE'''

'''LOAD MODEL'''
from keras.models import load_model
model = load_model(model_saving_path)


'''LOAD TEST DATA'''
normal_comment_test = get_data(vocab, stopwords, normal_test_path)
'''Normal comments have labels one'''
normal_labels_test = np.ones((len(normal_comment_test)), dtype = np.int) 

sara_comment_test = get_data(vocab, stopwords, sara_test_path)
'''Sara comments have labels zero'''
sara_labels_test = np.zeros(len(sara_comment_test), dtype = np.int)

X_test = np.concatenate((normal_comment_test, sara_comment_test), axis = 0)
y_test = np.concatenate((normal_labels_test, sara_labels_test), axis = 0)

'''Use model to predict'''
predicted = model.predict(X_test)

'''Select average of train label value tobe threshold'''
threshold = np.mean(y_train)   
print("Threshold = %f"%threshold)     
y_predicted = to_binary(predicted, threshold)


'''VALIDATION PHASE'''
from sklearn import metrics
print(metrics.classification_report(y_test, y_predicted, target_names=['Sara', 'Normal']))