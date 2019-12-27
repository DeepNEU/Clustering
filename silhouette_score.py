from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
data=pd.read_csv('929load.csv',nrows=4848,index_col=0)
data=data.drop(columns=['sum'])
classify=pd.read_csv('result.csv',nrows=4800,index_col=0)
classify=np.array(classify)
x=[]
y=[]
sum=0   
for i in range(4800):
    sample=data[i:i+48]
    sample=np.array(sample).reshape(929, 48)
    y_label=classify[i]       
    score= metrics.silhouette_score(sample,y_label)
    sum=score+sum
    y.append(score)
    print(i)
avescore=sum/4800
y.append(avescore)
df=pd.DataFrame(y,columns=['score'])
df.to_csv("轮廓系数Denseresult-3.csv")
print("success")

