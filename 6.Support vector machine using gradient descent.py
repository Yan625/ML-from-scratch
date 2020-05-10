#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def single_gradient(x_train,y_train, learning_rate,m):  
    m_slope = [0 for i in range(x_train.shape[1])]
    C=1
    Lambda=1/C
    M = x_train.shape[0]
    N = x_train.shape[1]
    for j in range(N):
        for i in range(M):
            x = x_train[i]
            y = y_train[i]
            y = y = np.where(y == 0, -1, 1)
            a=np.dot(m,x)
            if y*a>=1:
                m_slope[j] += 2*Lambda*m[j]/M
            else:
                m_slope[j] += -y*x[j]/M + 2*Lambda*m[j]/M
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
    C=1
    M = x_train.shape[0]
    for i in range(M):
        x = x_train[i]
        y = y_train[i]
        a = np.dot(m,x)
        sum1=0
        for i in m:
            sum1+=i**2
        list1=[0,1-y*a]
        sort=sorted(list1)
        total_cost += (C*sort[1]+1/2*sum1)/M
    return total_cost

def fit(x_train,y_train):
    learning_rate =0.001
    num_iterations = 100
    m = gd(x_train,y_train, learning_rate, num_iterations)
    return m

def predict(x_test):
    m=fit(x_train,y_train)
    y_predict=[]
    for x in x_test:
        if np.dot(m,x)>0:
            y_predicted=1
        else:
            y_predicted=0
        y_predict.append(y_predicted)
        
    return np.array(y_predict)

def score(y_predict,y_test):
    diff=y_predict-y_test
    sum1=0
    for i in diff:
        if i!=0:
            sum1+=1
    accuracy=1-(sum1/len(y_test))
    return accuracy

