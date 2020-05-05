#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#method 1: direct method

def fit(x_train,y_train):
    numerator=(x_train*y_train).mean()-x_train.mean()*y_train.mean()
    denominator=(x_train**2).mean()-x_train.mean()**2
    m=numerator/denominator
    c=y_train.mean()-m*x_train.mean()
    return m,c

def predict(x,m,c):
    return m*x+c

def score(y_true,y_predict):
    u=((y_true-y_predict)**2).sum()
    v=((y_true-y_true.mean())**2).sum()
    return 1-u/v

def cost(x,y,m,c):
    return ((y-(m*x+c))**2).mean()





#method 2: gradient descent

def single_gradient(data, learning_rate, m , c):
    m_slope = 0
    c_slope = 0
    M = len(data)
    for i in range(M):
        x = data[i, 0]
        y = data[i, 1]
        m_slope += (-2/M)* (y - m * x - c)*x
        c_slope += (-2/M)* (y - m * x - c)
        #print(m_slope,c_slope)
    new_m = m - learning_rate*m_slope
    new_c = c - learning_rate*c_slope
    return new_m, new_c

def gd(data, learning_rate, num_iterations):
    m = 0
    c = 0
    for i in range(num_iterations):
        m, c = single_gradient(data, learning_rate, m , c)
        print(i, " Cost: ", cost(data, m, c))
    return m, c

def cost(data, m, c):
    total_cost = 0
    M = len(data)
    for i in range(M):
        x = data[i, 0]
        y = data[i, 1]
        total_cost += (1/M)*((y - m*x - c)**2)
    return total_cost

def run():
    data = np.loadtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    num_iterations = 100
    m, c = gd(data, learning_rate, num_iterations)
    print(m, c)

