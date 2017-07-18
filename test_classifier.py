# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:47:00 2017

@author: levi
"""


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np
import codecs

dict_path =  "/home/levi/vi/challenges/cmt-data/dicts/id_full.txt"
stopwords_path = "/home/levi/vi/challenges/cmt-data/dicts/stop_words.txt"

normal_train_path = "/home/levi/vi/challenges/cmt-data/training_data/normal_comments.txt"
sara_train_path = "/home/levi/vi/challenges/cmt-data/training_data/sara_comments.txt"

normal_test_path = "/home/levi/vi/challenges/cmt-data/test_data/nornal_comments.txt"
sara_test_path = "/home/levi/vi/challenges/cmt-data/test_data/sara_comments.txt"

model_saving_path = "/home/levi/vi/challenges/cmt-data/sara_model.h5"

min_freq = 2
start_index = 3

min_comment_length = 3
max_comment_length = 50

out_of_vocab = 2
end_line = 1

min_freq = 2
start_index = 3

min_comment_length = 3
max_comment_length = 50

def get_stopwords():    
    stopwords = set()
    f = codecs.open(stopwords_path, 'r', encoding='utf8')    
    for line in f:
        stopwords.add(line.strip())    
    f.close()    
    return stopwords

def get_vocab():
    vocab = {}
    vocab['.'] = 1
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

def get_data(vocab, stopwords, path):
    f = codecs.open(path, 'r', encoding='utf8')    
    data = []
    for line in f:
        line = line.lower().replace('.', ' . ')
        tokens = line.split()        
        encoded = [1]        
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
    

stopwords = get_stopwords()
vocab = get_vocab()
    
normal_comments = get_data(vocab, stopwords, normal_test_path)
sara_comments = get_data(vocab, stopwords, sara_test_path)
# LSTM for sequence classification

from keras.models import load_model

model = load_model(model_saving_path)
# Final evaluation of the model

normal_predict = model.predict(normal_comments)
sara_predict = model.predict(sara_comments)


normal_mean = 0.902193
sara_mean = 0.402668
all_mean = 0.800670737425
thread_hold = 0.8

def greater_than(A, t):
    s = 0
    for a in A:
        if a > t:
            s += 1
    return s

def less_than(A, t):
    s = 0
    for a in A:
        if a < t:
            s += 1
    return s

true_normal = greater_than(normal_predict, thread_hold)
true_sara = less_than(sara_predict, thread_hold)
false_normal = len(normal_predict) - true_normal
false_sara = len(sara_predict) - true_sara

sara_presision = true_sara / (true_sara + false_normal)
sara_recall = true_sara / len(sara_predict)
accuracy = (true_normal + true_sara )/ (len(normal_predict) + len(sara_predict)) 
f1 = 2 * sara_presision * sara_recall / (sara_presision + sara_recall)

print("Sara Precision: %f"%sara_presision)
print("Sara Recall: %f"%sara_recall)
print("Accuracy: %f"%accuracy)
print("F1: %f"%f1)
