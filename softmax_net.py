# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 


saved_model = "D:/Data/vi/python/model.npy"

l0 = 784
l1 = 20
l2 = 10

bias0 = 1
bias1 = 1

W01 = np.zeros((l1, l0+1), dtype=np.float32)
W12 = np.zeros((l2, l1+1), dtype=np.float32)

D01 = np.zeros((l1, l0+1), dtype=np.float32)
D12 = np.zeros((l2, l1+1), dtype=np.float32)

A0 = np.zeros((l0), dtype=np.float32)
A1 = np.zeros((l1), dtype=np.float32)
A2 = np.zeros((l2), dtype=np.float32)


def relu(Z):   
    R_Z = np.array([ max(i, 0) for i in Z])
    return R_Z

def softmax(Z):
    z_max = np.max(Z)
    Z_reduce = Z - z_max
    E_Z = np.exp(Z_reduce)
    Sum_E_Z = np.sum(E_Z)
    return E_Z / Sum_E_Z

def input_data(X):
    global A0
    for i in range(l0):
        A0[i] = X[i]
    return

def forward_propagate(X):
    input_data(X)
    global A1, A2
    A0_Bias = np.concatenate(([bias0], A0))
    A1 = np.dot(W01, A0_Bias)
    A1 = relu(A1)
    A1_Bias = np.concatenate(([bias1], A1))
    A2 = np.dot(W12, A1_Bias)
    A2 = softmax(A2)
    return

X = np.random.randn(l0)

forward_propagate(X)

print(X)
print(A0)
print(A1)
print(A2)

def backward_propagate(Y):
    global D01, D12
    Delta2 = A2 - Y
    W12_T = np.transpose(W12)
    Delta1 = np.dot(W12_T, Delta2)
    for i in range(1, l1+1):
        for j in range(l2+1):
            D12[i, j] = A1[i] *1
    
    return













































    
    
 
