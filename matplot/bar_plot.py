#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:13:52 2019

@author: chance
"""

import matplotlib.pyplot as plt
import seaborn as sns

# data prepare

x =['cat'+str(i) for i in range(1,6)]
y =[5,4,8,12,7]

# use matplot
plt.bar(x,y)
plt.show()

# use seaborm

sns.barplot(x,y)
plt.show()