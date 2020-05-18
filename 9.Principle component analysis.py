#!/usr/bin/env python
# coding: utf-8

# # PCA

# In[ ]:


def fit(data):
    cov=np.cov(data.T)
    eig_values,eig_vectors=np.linalg.eig(cov)
    eig_value_vector_pair=[]
    for i in range(len(eig_values)):
        eig_vec=eig_vectors[:,i]
        eig_value_vector_pair.append((eig_values[i],eig_vec))
    eig_value_vector_pair.sort(reverse=True)
    return eig_value_vector_pair

def transform(x):
        a=eig_value_vector_pair[0:k]
        eig_vector=[]
        for i in a:
            eig_vector.append(i[1])
        x = x - x.mean
        return np.dot(x, eig_vector.T)

