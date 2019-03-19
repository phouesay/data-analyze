#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:35:55 2019

@author: chance
"""
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

data = pd.read_csv('./data/data.csv',encoding='gbk')
# print(data.info())
# print(data.shape)
train_x = data[['2019年国际排名','2018世界杯','2015亚洲杯']]
df =pd.DataFrame(train_x)
model = KMeans(n_clusters=3)
# 规范化到【0，1】空间
min_max_scaler = MinMaxScaler()
train_x = min_max_scaler.fit_transform(train_x)
model.fit(train_x)
predict_y = model.predict(train_x)
# merge
result = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
result.rename({0:u'聚类'},axis=1,inplace=True)
print(result)