#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:33:58 2019

@author: chance
"""

from sklearn import svm
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


# 显示全列
pd.set_option('display.max_columns',None)

# model = svm.SVC(kernel='rbf',C=1.0,gamma='auto')
datas = pd.read_csv("./data/data.csv")

# explore
#print(datas.columns)
#print(datas.head(5))
#print(datas.describe())

# devide groups
features_mean = list(datas.columns[2:12])
features_mean =list(datas.columns[12:22])
features_worst =list(datas.columns[22:32])

# 清洗
datas.drop('id',axis=1,inplace = True)
# B,M map to 0,1
datas['diagnosis'] = datas['diagnosis'].map({'M':1,"B":0})

#sns.countplot(datas['diagnosis'],label='Count')
#plt.show()

# heatmap 呈现相关性
corr = datas[features_mean].corr()
plt.figure(figsize=(14,14))
# annot =True，显示每个方格的数据
#sns.heatmap(corr,annot=True)
#sns.show()
# extract feature
# 特征选择
features_remain = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean'] 
# split train and test
train,test =train_test_split(datas,test_size=0.3)
train_x = train[features_mean]
train_y=train["diagnosis"]
test_x = test[features_mean]
test_y = test["diagnosis"]
# 规范化,采用Z-score规范化数据，保证每个特征维度的数据均值为0，方差为1
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.transform(test_x)
# predict
svm_model = svm.SVC()
svm_model.fit(train_x,train_y)
prediction = svm_model.predict(test_x)
score = metrics.accuracy_score(prediction,test_y)
print("准确率为: ",score)




