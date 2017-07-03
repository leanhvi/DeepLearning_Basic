#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:11:37 2017

@author: levi
"""

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
import numpy as np
import mnist


labels = list(mnist.read_labels(dataset='training', path='mnist-data'))
images = list(mnist.read_images(dataset='training', path='mnist-data'))


#utility functions region
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


#test region


#training region
def train():
    target = make_target(labels)
    predictors = np.array(images)
    print(predictors.shape)
    
    n_cols = predictors.shape[1]
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape = (n_cols,)))
    model.add(Dense(10, activation='softmax'))
     
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'] )
    
    model.fit(predictors, target)
    return model





test_images = list(mnist.read_images(dataset='testing', path='mnist-data'))
test_predic = np.array(test_images)

test_data = list(mnist.read(dataset='testing', path='mnist-data'))




model = train()
#model.save('models/mnist.h5')

#model = load_model('models/mnist.h5')

for i in range(10, 20):
    to_predic= np.reshape(test_predic[i], (1, 784))
    array = model.predict(to_predic)
    print(tonum(array))
    mnist.show(test_data[i][1])














