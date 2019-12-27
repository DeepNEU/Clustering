#计算模型分类结果的DBI值
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
data=pd.read_csv("929loadchange.csv")
for j in range(1):
    y=[]
    k=2+j
    sum=0
    classify=pd.read_csv('result4.csv',index_col=0)
    classify=np.array(classify)
    print(classify.shape)
    print(classify[0])
    for i in range(23088,25680):
            sample=data[i:i+48]
            sample=np.array(sample).reshape(929, 48)
            #y_pred = KMeans(n_clusters=k).fit(sample)       
            score= metrics.davies_bouldin_score(sample,classify[i-23088])
            print(score)
            y.append(score)
            sum=sum+score
    avescore=sum/2592
    y.append(avescore)
    df=pd.DataFrame(y,columns=['score'])
    df.to_csv("DBI4.csv")
