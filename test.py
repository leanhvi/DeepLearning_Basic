
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np


dict_path =  "/home/levi/vi/challenges/cmt-data/dicts/id_full.txt"
stopwords_path = "/home/levi/vi/challenges/cmt-data/dicts/stop_words.txt"

normal_train_path = "/home/levi/vi/challenges/cmt-data/training_data/normal_comments.txt"
sara_train_path = "/home/levi/vi/challenges/cmt-data/training_data/sara_comments.txt"

normal_test_path = "/home/levi/vi/challenges/cmt-data/test_data/nornal_comments.txt"
sara_test_path = "/home/levi/vi/challenges/cmt-data/test_data/sara_comments.txt"

min_freq = 10
start_index = 3

max_comment_length = 500

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
            print("Encounter a stopword %s:" % token)
            continue
        vocab[token] = index
        index += 1        
    f.close()
    return
    
#get_stopwords()
#get_vocab()

def get_data(path):
    f = open(path, 'r')    
    data = []

    for line in f:
        tokens = line.split()        
        encoded = []
        for token in tokens:
            
    
    f.close
    return    
    
    


