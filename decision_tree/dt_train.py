#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:24:42 2019

@author: chance
"""
import pandas as pd
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score

#pd.set_option('display.width',None)
#pd.set_option('display.max_columns',10)
#pd.set_option("display.max_colwidth",100)

train_data =pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")
# data explore
print(train_data.info())
print('-'*30)
print(train_data.shape)
print('-'*30)
print(len(train_data))
print('-'*30)
print(train_data.columns)
print('-'*30)
print(train_data.describe())
print('-'*30)
# print(train_data.describe(include=['0']))
print('-'*30)
print(train_data.head())
print('-'*30)
print(train_data.tail())
# deal nan
train_data["Age"].fillna(train_data['Age'].mean(),inplace= True)
test_data["Age"].fillna(test_data['Age'].mean(),inplace=True)

train_data["Fare"].fillna(train_data['Fare'].mean(),inplace= True)
test_data["Fare"].fillna(test_data['Fare'].mean(),inplace=True)

# print(train_data['Embarked'].value_counts())
train_data['Embarked'].fillna('S',inplace=True)
test_data["Embarked"].fillna("S",inplace= True)

# extract features
features =[ 'Pclass','Sex', 'Age', 'SibSp','Parch','Fare', 'Embarked']
train_features =train_data[features]
train_labels =train_data['Survived']
test_features =test_data[features]

# 特征向量转特征值矩阵
dvec = DictVectorizer(sparse=False)
train_features=dvec.fit_transform(train_features.to_dict(orient="record"))
#print(train_features)
#sys.exit()
# print(dvec.feature_names_)
# choose model
clf = DecisionTreeClassifier(criterion="entropy")
# DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
#            max_features=None, max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
#            splitter='best')

clf.fit(train_features,train_labels)

test_features = dvec.transform(test_features.to_dict(orient='record'))
pred_labels = clf.predict(test_features)

acc_decision_tree = round(clf.score(train_features,train_labels),6)
print(u"score 准确率为 %.4lf"% acc_decision_tree)
print("-"*30)
# k 折交叉验证
print(u"cross_val_score 准确率为 %.4lf "% np.mean(cross_val_score(clf,train_features,train_labels,cv=10)))




