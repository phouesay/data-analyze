#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:32:20 2019

@author: chance
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import sys

documents = [
    'this is the bayes document',
    'this is the second second document',
    'and the third one',
    'is this the document'
]

def load_stop_words(filePath):
    stop_words=[]
    with open(filePath,'r') as tt:
        for line in tt.readlines():
            stop_words.append(line.strip())
    return stop_words
            
file_path = "./data/stop_word.txt"
 
stop_words = load_stop_words(file_path)

tfidf_vec = TfidfVectorizer(stop_words=stop_words,max_df=0.5)

features = tfidf_vec.fit_transform(documents)
print(features)
sys.exit()

trans_feature,test_feature,train_label,test_label = train_test_split()

print('不重复的词：',tfidf_vec.get_feature_names())
print("id of words",tfidf_vec.vocabulary_)
print("tf-idf of words",features.toarray())
# alpha 做平滑处理
clf = MultinomialNB(alpha=0.001).fit(trans_feature,train_label)

# tst_tf = TfidfVectorizer()
predicted_lables = clf.predict(test_feature)

score = metrics.accuracy_score(test_label,predicted_lables)

print("NB test 正确率为.4lf"%score)
 