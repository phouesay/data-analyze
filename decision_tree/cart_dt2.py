#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:57:12 2019

@author: chance
"""

# encoding=uft-8
# CART 回归树
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
# data prepare
boston =load_boston()
# explore data
#print(boston.feature_names)
# get features and labels
features = boston.data
prices = boston.target
# divide data set
train_features,test_features,train_prices,test_prices = train_test_split(features,prices,test_size=0.33)
# create cart
dtr = DecisionTreeRegressor()
# fit
dtr.fit(train_features,train_prices)
# predict
predict_prices = dtr.predict(test_features)
# 测试集结果评测
print("回归树二乘偏差均值：",mean_squared_error(test_prices,predict_prices))
print("回归树绝对值便缠均值：",mean_absolute_error(test_prices,predict_prices))
