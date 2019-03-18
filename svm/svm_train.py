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

sns.countplot(datas['diagnosis'],label="Count")
plt.show()
