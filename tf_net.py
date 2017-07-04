# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 09:50:12 2017

@author: levi
"""

import time
import numpy as np
import tensorflow as tf

l0 = 784
l1 = 20
l2 = 10

bias0 = tf.constant([[1]], dtype=tf.float32)
bias1 = tf.constant([[1]], dtype=tf.float32)

W01 = tf.Variable(tf.ones((l1, l0+1), dtype=tf.float32))
W12 = tf.Variable(tf.ones((l2, l1+1), dtype=tf.float32))

D01 = tf.Variable(tf.zeros((l1, l0+1), dtype=tf.float32))
D12 = tf.Variable(tf.zeros((l2, l1+1), dtype=tf.float32))

A0 = tf.Variable(tf.zeros([l0, 1], dtype=tf.float32))
A1 = tf.Variable(tf.zeros([l1, 1], dtype=tf.float32))
A2 = tf.Variable(tf.zeros([l2, 1], dtype=tf.float32))

#init = tf.global_variables_initializer()
#sess = tf.Session()
#sess.run(init)
#s = sess.run(A2)
#print(s)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

def forward_propagate(X):
    global A0, A1, A2
    
    A0 = tf.constant(X, dtype=tf.float32)
    
    A0_Bias = tf.concat([bias0, A0], 0)

    A1 = tf.matmul(W01, A0_Bias)
    
    A1 = tf.nn.relu(A1)
    A1_Bias = tf.concat([bias1, A1], 0)

    A2 = tf.matmul(W12, A1_Bias)
        
    A2 = tf.nn.softmax(A2)
    
    return A2    


def backward_propagate(Y):
    global D01, D12
    
    Delta2 = A2 - tf.constant(Y, dtype=tf.float32)    
    A1_Bias = tf.concat([bias1, A1], 0)

    D21 = tf.matmul(A1_Bias, Delta2)
    D12 = D12 + tf.transpose(D21)
    
    Delta1 = tf.matmul(W12, Delta2, transpose_a=True)            
    A0_Bias = tf.concat([bias0, A0], 0)
    D10 = tf.matmul(A0_Bias, Delta1)
    D01 = D01 + tf.transpose(D10)
                
    return D01








    

    
import mnist

def toarray(k):
    array = np.zeros(shape=(10))
    array[k] = 1
    return array

def tonum(array):    
    return np.argmax(array)        

def make_target(data):
    lst = []
    for i in data:
        lst.append(toarray(i))
    return np.array(lst)

images = list(mnist.read_images(dataset='training', path='mnist-data'))

labels = list(mnist.read_labels(dataset='training', path='mnist-data'))

targets = make_target(labels)
    
    
    
#X = np.array(images[1]).reshape(l0, 1)
#
#fw = forward_propagate(X)
#
#print(X)
#print(sess.run(fw))
    
Y = targets[0].reshape(l2, 1)    

Delta2 = A2 - tf.constant(Y, dtype=tf.float32)    
A1_Bias = tf.concat([bias1, A1], 0)

D21 = tf.matmul(A1_Bias, Delta2, transpose_b=True)
print(sess.run(D21))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    