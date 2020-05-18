#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class K_Means:
    def __init__(self, k = 2, max_iter = 100):
        print("constructor")
        self.k = k
        self.max_iter = max_iter
        
    def fit(self,data):
        self.means=[]
        #randomly initialize the means
        for i in range(self.k):
            self.means.append(data[i])
        for i in range(self.max_iter):
            #assign the data point to the clusters that they belong to 
            #creaate empty clusters
            clusters=[]
            for j in range(self.k):
                clusters.append([])
            for point in data:
                #find distance to all the mean values
                distances=[((point-m)**2).sum() for m in self.means]
                #find the min distance
                minDistance=min(distances)
                #find the mean for which we got the minimum distance-->l
                l=distances.index(minDistance)
                #add this point to cluster l
                clusters[l].append(point)


            #calculate the new mean values
            change=False
            for j in range(self.k):
                new_mean=np.average(clusters[j],axis=0)
                if not np.array_equal(self.means[j],new_mean):
                    change=True
                self.means[j]=new_mean
            if not change:
                break
      
    
    def predic(self,test_data):
        predictions=[]
        for point in test_data:
            distances=[((point-m)**2).sum() for m in self.means]
            minDistance=min(distances)
            l=distances.index(minDistance)
            predictions.append(l)
        return predictions

