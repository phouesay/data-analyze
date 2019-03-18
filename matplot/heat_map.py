#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:26:32 2019

@author: chance
"""

import matplotlib.pyplot
import seaborn as sns

# data prepare
# data set in seaborn

#      year      month  passengers
# 0    1949    January         112
# 1    1949   February         1s18
# 2    1949      March         132
# 3    1949      April         129

flights = sns.load_dataset("flights")
data = flights.pivot("year","month","passengers")
# draw
sns.heatmap(data)
plt.show()
