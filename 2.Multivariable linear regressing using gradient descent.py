#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def single_gradient(x_train,y_train, learning_rate,m):  
    m_slope = [0 for i in range(x_train.shape[1])]
    M = x_train.shape[0]
    N = x_train.shape[1]
    for j in range(N):
        for i in range(M):
            x = x_train[i]
            y = y_train[i]
            a = np.dot(m,x)
            m_slope[j] += (-2/M)* (y - a)*x[j]
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
        total_cost += (1/M)*((y - a)**2)
    return total_cost

def run():
    learning_rate = 0.4
    num_iterations = 125
    m = gd(x_train,y_train, learning_rate, num_iterations)
    return m

def predict(x_test):
    m=run()
    y_predict=[]
    for i in range(len(x_test)):
        a=sum(x_test[i]*m)
        y_predict.append(a)
    return y_predict

def score(y_true,y_predict):
    u=sum((y_true-y_predict)**2)
    v=sum((y_true-y_true.mean())**2)
    return 1-u/v

