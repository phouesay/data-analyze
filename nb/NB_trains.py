#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:06:16 2019

@author: chance
"""

import os
import jieba
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


label_map ={'体育':0,'女性':1,'文学':2,'校园':3}

# 加载停用词表
def get_stop_word(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        stop_words =[line.strip() for line in f.readlines()]
    return stop_words

# 对文档进行分词
def load_data(base_path):
    documents =[]
    labels =[]
    for root,dirs,files in os.walk(base_path):
        # 会在该目录下读取到 .DS_Store 文件 
        for file in files:
            # 过滤
            if file.endswith('.DS_Store'):
                continue
            label = root.split('/')[-1]
            labels.append(label)
            file_name = os.path.join(root,file)
            #with open(file_name,'rb') as fs:
            with open(file_name,'r',encoding='gb18030') as fs:
                # str --encode--> bytes --decode--> str
                text = fs.read()
                textlist = list(jieba.cut(text))
                documents.append(" ".join(textlist))
               # documents.append(contentlist)
    return documents,labels

def train_fun(train_contents,train_labels,test_contents,test_labels,stop_words):
    # 计算单词的权重,得到特征空间
    tf = TfidfVectorizer(stop_words=stop_words,max_df=0.5)
    train_features= tf.fit_transform(train_contents)
    test_features = tf.transform(test_contents)
    
    # fit
    clf = MultinomialNB(alpha =0.01).fit(train_features,train_labels)
    # predict
    #test_tf = TfidfVectorizer(stop_words=stop_words,max_df =0.5,vocabulary=tf.vocabulary_)
    #test_features  = test_tf.fit_transform(test_contents)
    predicted_labels = clf.predict(test_features)
    
    return metrics.accuracy_score(test_labels,predicted_labels)
    
    
    
    

stop_words =get_stop_word('./data/stop/stopword.txt')
train_documents,train_labels = load_data('./data/train')
test_documents,test_labels = load_data('./data/test')
score = train_fun(train_documents,train_labels,test_documents,test_labels,stop_words)
print("NB 准确率是 ：",score)
# print(stop_words)

