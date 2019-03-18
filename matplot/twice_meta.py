#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:53:02 2019

@author: chance
"""

import matplotlib.pyplot as plt
import seaborn as sns

# data prepare
tips = sns.load_dataset("tips")
#print(tips)

#     total_bill   tip     sex smoker   day    time  size
#0         16.99  1.01  Female     No   Sun  Dinner     2
#1         10.34  1.66    Male     No   Sun  Dinner     3
#2         21.01  3.50    Male     No   Sun  Dinner     3

sns.jointplot(x="total_bill",y="tip",data =tips,kind="scatter")
# 核密度
sns.jointplot(x="total_bill",y="tip",data =tips,kind="kde")
sns.jointplot(x="total_bill",y="tip",data =tips,kind="hex")
sns.show()