# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 10:27:48 2017

@author: levi
"""

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 


dim = 784
batch_size = 100
n_classes = 10

W = np.zeros((dim, n_classes), dtype=np.float32)
G = np.zeros((dim, n_classes), dtype=np.float32)
A0 = np.zeros((batch_size, dim), dtype=np.float32) #Input layer
Z1 = np.zeros((batch_size, n_classes), dtype=np.float32) #Before activate output layer
A1 = np.zeros((batch_size, n_classes), dtype=np.float32) #Output layer

'''
A0: batch_size x dim
W:  dim x n_classes
Z1: batch_size x n_classes
A1: batch_size x n_classes


'''

def softmax(Z):
    A = []
    len0 = len(Z)
    for i in range(len0):
        max_value = np.max(Z[i])
        Z_i_reduced = Z[i] - max_value
        E_Z_i = np.exp(Z_i_reduced)
        sum_E_Z_i = np.sum(E_Z_i)
        SM_Z_i = E_Z_i / sum_E_Z_i
        A.append(SM_Z_i)
    return np.array(A)

def forward(X):
    global A0, Z1, A1
    A0 = X
    Z1 = np.dot(A0, W)
    A1 = softmax(Z1)        
    return
    
def backward(Y):
    global W, G
    
    return    





#import tensorflow as tf
#Zt = tf.constant([[1, 2], [5, 7]], dtype=tf.float32)
#At = tf.nn.softmax(Zt, dim=1)
#
#sess = tf.Session()
#
#model = sess.run(At)
#print(model)


    
    


    