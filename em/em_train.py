#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:33:04 2019

@author: chance
"""

from sklearn.mixture import GaussianMixture
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
import sys

pd.set_option("display.max_columns",None)
data = pd.read_csv('./data/heros.csv',encoding='gb18030')
features_data= data[data.columns[1:-2]]
 
# 设置 plt 正确显示中文
#plt.rcParams['font.family']=['sans-serif']
#plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['font.family'] = ['Arial Unicode MS'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

#plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
# heatmap 对非数值类数据不显示热力图
corr = data.corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr,annot=True)
plt.show()
# explore
#print(data.columns)

# 相关性大的属性保留一个，因此可以对属性进行降维
features_remain = [u'最大生命', u'初始生命', u'最大法力', u'最高物攻', u'初始物攻', u'最大物防', u'初始物防', u'最大每5秒回血', u'最大每5秒回蓝', u'初始每5秒回蓝', u'最大攻速', u'攻击范围']
data_remain = data[features_remain]
data_remain[u'最大攻速']=data_remain[u'最大攻速'].apply(lambda x:float(x.strip('%'))/100)
data_remain[u'攻击范围']=data_remain[u'攻击范围'].map({'近战':0,'远程':1})
# 采用Z-score 规范化数据
ss = StandardScaler()
data_new = ss.fit_transform(data_remain)

# model and fix 
gmm= GaussianMixture(n_components=30,covariance_type='full')
gmm.fit(data_new)
# predict
prediction = gmm.predict(data_new)
# print(prediction)
data.insert(0,'分组',prediction)
data.to_csv('./data/heros_out.csv',index=False,sep=',',encoding='utf_8_sig')

