
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
data=pd.read_csv("E:/learn/929loadchange.csv",nrows=4800)
data=np.array(data)
data=data.reshape(929,4800)
print(data)
print(data.shape)
#计算kmeans分类结果的DBI
for j in range(1):
    y=[]
    k=2+j
    sum=0
    kmeans = KMeans(n_clusters=4, random_state=1).fit(data)
    classify = kmeans.labels_
    print(classify)
print(classify.shape)
df=pd.DataFrame(classify)
df.to_csv("E:/learn/4-Kmeans.csv")
       
score= metrics.davies_bouldin_score(data,classify)
print("score",score)



#Max-AC分类与计算DBI评分
import pandas as pd
from sklearn.cluster import k_means
import numpy as np
import random

def MAX_AC(X,h):
    average = np.mean(X)
    var = np.var(X)*len(X)
    sum = 0
    for i in range(len(X)-h):
        sum = sum + (X[i]-average)*(X[i+h]-average)/var
    return sum


def greedyClustering(k,X,h):
    classify=[[] for i in range(k)]
    classifyIndex = [1 for i in range(k)]
    result = [-1 for i in range(len(X))]
    #取出随机的k个值
    rs = random.sample(range(0,len(X)),k)
    #把随机取出来的按index先分类
    X = np.array(X)
    for i in range(k):
        result[rs[i]] = i
        classify[i] = X[rs[i]]
    for i in range(len(X)):
        if i in rs:
            continue
        else:
            temp = X[i]
            maxDistance = [0 for a in range(k)]
            index = 0
            for j in range(k):
                tempMAX = MAX_AC((temp + classify[j])/(classifyIndex[j] + 1),h)
                MAX = MAX_AC(classify[j]/classifyIndex[j],h)
                maxDistance[j] = tempMAX - MAX
            index = maxDistance.index(max(maxDistance))
            classify[index] = classify[index] + temp
            classifyIndex[index] = classifyIndex[index] + 1
            result[i] = index
    return result;



y_pred = greedyClustering(4,data,5)
print(y_pred)
score= metrics.davies_bouldin_score(data,y_pred)
print("score",score)
