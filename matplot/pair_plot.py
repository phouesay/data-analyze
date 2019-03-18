#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:57:49 2019

@author: chance
"""

import matplotlib.pyplot as plt
import seaborn as sns

# data prepare
iris = sns.load_dataset("iris")
# use seaborn
sns.pairplot(iris)
sns.show(s)