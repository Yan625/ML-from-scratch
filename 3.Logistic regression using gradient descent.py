#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
import math

def sigmoid(x):
        return 1 / (1 + np.exp(-x))


def single_gradient(x_train,y_train, learning_rate,m):  
    m_slope = [0 for i in range(x_train.shape[1])]
    M = x_train.shape[0]
    N = x_train.shape[1]
    for j in range(N):
        for i in range(M):
            x = x_train[i]
            y = y_train[i]
            a=np.dot(m,x)
            m_slope[j] += (-1/M)* (y - sigmoid(a))*x[j]
    m_slope=np.array(m_slope)
    new_m=m-m_slope*learning_rate   
    return new_m
            
def gd(x_train,y_train, learning_rate, num_iterations):
    m = [0 for i in range(x_train.shape[1])]
    for i in range(num_iterations):
        m = single_gradient(x_train,y_train,learning_rate,m)
        #print(i, " Cost: ", cost(x_train,y_train,m))
    return m

def cost(x_train,y_train, m):
    total_cost = 0
    M = x_train.shape[0]
    for i in range(M):
        x = x_train[i]
        y = y_train[i]
        a = np.dot(m,x)
        total_cost += (1/M)*(-y*math.log(sigmoid(a))-(1-y)*math.log(1-sigmoid(a)))
    return total_cost

def fit(x_train,y_train):
    learning_rate =4
    num_iterations = 16
    m = gd(x_train,y_train, learning_rate, num_iterations)
    return m

def prob(x_test):
    m=fit(x_train,y_train)
    Prob=[]
    for i in range(len(x_test)):
        x=x_test[i]
        a=np.dot(m,x)
        p = sigmoid(a)
        Prob.append(p)
    return Prob

def predict(x_test):
    Prob=prob(x_test)
    y_predicted = [1 if i > 0.6 else 0 for i in Prob]
    return np.array(y_predicted)

def accuracy(x_test):
    a=y_predict-y_train
    count=0
    for i in a:
        if i==0:
            count+=1
    accuracy=count/len(a)
    return accuracy 

