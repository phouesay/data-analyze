#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:45:48 2019

@author: chance
"""

# encoding =uft-8
# CART 分类树
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# data prepare
iris = load_iris()
# get feature set and labels
features = iris.data
labels = iris.target
# divide test data set and train set
train_features,test_features,train_labels,test_labels = train_test_split(features,labels,test_size=0.3,random_state=0)
# create cart
clf = DecisionTreeClassifier(criterion='gini')
# fit
clf = clf.fit(train_features,train_labels)
# prediction
test_predict= clf.predict(test_features)
# accuracy
score = accuracy_score(test_labels,test_predict)

print("cart 分类树准确率 %.4lf" % score)
