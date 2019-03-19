#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 08:19:34 2019

@author: chance
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import metrics
import sys
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsRegressor

# prepare data
digits = load_digits()
data = digits.data
# explore
print(data.shape)
print(digits.images[0])
print(digits.target[0])

#plt.gray()
#plt.imshow(digits.images[0])
#plt.show()
train_x,test_x,train_y,test_y = train_test_split(data,digits.target,test_size=0.25,random_state=33)


# Z-score规范化
ss = StandardScaler()
train_ss_x = ss.fit_transform(train_x)

test_ss_x = ss.transform(test_x)

model = KNeighborsClassifier()
model.fit(train_ss_x,train_y)
prediction = model.predict(test_ss_x)
score = metrics.accuracy_score(prediction,test_y)
print("KNN  准确率：",score)

svm = SVC()
svm.fit(train_ss_x,train_y)
predict_y = svm.predict(test_ss_x)
svm_score = metrics.accuracy_score(predict_y,test_y)
print("SVM 的准确率：",svm_score)

# 采用 min-max 规范化
mm = MinMaxScaler()
train_mm_x = mm.fit_transform(train_x)
test_mm_x = mm.transform(test_x)
mnb = MultinomialNB()
mnb.fit(train_mm_x,train_y)
mnb_predict_y = mnb.predict(test_mm_x)
mnb_score = metrics.accuracy_score(mnb_predict_y,test_y)
print("多项式朴素贝叶斯 准确率:",mnb_score)

dtc= DecisionTreeClassifier()
dtc.fit(train_mm_x,train_y)
dtc_predict = dtc.predict(test_mm_x)
dtc_score = metrics.accuracy_score(dtc_predict,test_y)
print("CART 决策树准确率",dtc_score)



