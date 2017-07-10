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

train_size = 60000;  
batch_size = 2;  
lr = 1

bias0 = tf.ones([1, batch_size], dtype=tf.float32)
bias1 = tf.ones([1, batch_size], dtype=tf.float32)

W01 = tf.Variable(tf.zeros((l1, l0+1), dtype=tf.float32))
W12 = tf.Variable(tf.zeros((l2, l1+1), dtype=tf.float32))

D01 = tf.Variable(tf.zeros((l1, l0+1), dtype=tf.float32))
D12 = tf.Variable(tf.zeros((l2, l1+1), dtype=tf.float32))

A0 = tf.Variable(tf.zeros([l0, batch_size], dtype=tf.float32))
A1 = tf.Variable(tf.zeros([l1, batch_size], dtype=tf.float32))
A2 = tf.Variable(tf.zeros([l2, batch_size], dtype=tf.float32))
Z1 = tf.Variable(tf.zeros([l1, batch_size], dtype=tf.float32))
Z2 = tf.Variable(tf.zeros([l2, batch_size], dtype=tf.float32))


def forward_propagate(X):
    global A0, A1, A2
    
    A0 = tf.constant(X, dtype=tf.float32)
    
    A0_Bias = tf.concat([bias0, A0], 0)

    Z1 = tf.matmul(W01, A0_Bias)
    
    A1 = tf.nn.relu(Z1)
    A1_Bias = tf.concat([bias1, A1], 0)

    Z2 = tf.matmul(W12, A1_Bias)
        
    A2 = tf.nn.softmax(Z2, dim = 0)
    
    return   


def backward_propagate(Y):
    global D01, D12
    
    Delta2 = (A2 - tf.constant(Y, dtype = tf.float32))
    A1_Bias = tf.concat([bias1, A1], 0)

    D12 += tf.matmul(Delta2, A1_Bias, transpose_b = True)
    
    Delta1 = tf.matmul(W12, Delta2, transpose_a = True)            
    Delta1_slice = tf.slice(Delta1, [1, 0], [l1, batch_size])   
    
    A0_Bias = tf.concat([bias0, A0], 0)
    D01 += tf.matmul(Delta1_slice, A0_Bias, transpose_b = True)
                
    return

def update_weights():
    global W01, W12
    W01 += lr * D01
    W12 += lr * D12
    return
    
    
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
    


start = time.time()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

n_batchs = train_size / batch_size
X = np.array(images).reshape(n_batchs, l0, batch_size)
Y = targets.reshape(n_batchs, l2, batch_size) 

for i in range(1):    
    forward_propagate(X[i])
    backward_propagate(Y[i])
    
    #update_weights()        
    print("Iter: ", i)    
    
    
    
print(sess.run(tf.Print(D12, [D12])))
print("Time: ", time.time() - start)


#x = tf.ones([batch_size, 1], dtype=tf.float32)
#print(sess.run(x))

#print(X[0])
#print(Y[0])



    

    