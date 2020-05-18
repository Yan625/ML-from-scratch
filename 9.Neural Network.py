#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def sig(z):
    return 1/(1 + np.exp(-z))

def derivativeSig(z):
    return sig(z)*(1 - sig(z))





#without hidden layer
import numpy as np
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0,0,0,1]]).T

weights = 2* np.random.random((2, 1)) - 1
bias = 2 * np.random.random(1) - 1
lr = 0.1

for iter in range(10000):
    output0 = X
    output = sig(np.dot(output0, weights) + bias)

    first_term = output - Y
    input_for_last_layer = np.dot(output0, weights) + bias
    second_term = derivativeSig(input_for_last_layer)
    first_two = first_term * second_term
    first_two.shape

    changes = np.dot(output0.T, first_two)
    weights = weights - lr*changes
    bias_change = np.sum(first_two)
    bias = bias - lr * bias_change
output = sig(np.dot(X, weights) + bias)







#with one hidden laye
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0,1,1,0]]).T

wh = 2* np.random.random((2, 2)) - 1 
bh = 2* np.random.random((1, 2)) - 1 
wo = 2 * np.random.random((2, 1)) - 1
bo = 2 * np.random.random((1, 1)) - 1
lr = 0.1

for iter in range(10000):
    output0 = X
    inputHidden=np.dot(output0, wh) + bh
    outputHidden = sig(inputHidden)
    inputForOutputLayer=np.dot(outputHidden, wo) + bo
    output = sig(inputForOutputLayer)

    first_term_output_layer = output-Y
    second_term_output_layer = derivativeSig(inputForOutputLayer)
    first_two_output_layer = first_term_output_layer*second_term_output_layer

    first_term_hidden_layer = np.dot(first_two_output_layer,wo.T)
    second_term_hidden_layer = derivativeSig(inputHidden)
    first_two_hidden_layer =  first_term_hidden_layer*second_term_hidden_layer

    changes_output = np.dot(outputHidden.T,first_two_output_layer)
    changes_output_bias = np.sum(first_two_output_layer,axis=0,keepdims=True)

    changes_hidden = np.dot(output0.T,first_two_hidden_layer)
    changes_hidden_bias = np.sum(first_two_hidden_layer,axis=0,keepdims=True)

    wo=wo-lr*changes_output
    bo=bo-lr*changes_output_bias

    wh=wh-lr*changes_hidden
    bh=bh-lr*changes_hidden_bias

output0 = X
inputHidden=np.dot(output0, wh) + bh
outputHidden = sig(inputHidden)
inputForOutputLayer=np.dot(outputHidden, wo) + bo
output = sig(inputForOutputLayer)


# turn the learning rate and number of iterations to achieve better results

