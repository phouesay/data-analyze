#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:16:54 2019

@author: chance
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# data prepare
# gen 10*4 dimensions data between 0 and 1,such as 
# [[-1.28513059 -1.55849067  1.17281365 -0.24583337]
# [ 0.40781531  1.4084505   0.50122232  1.03652482]
# [ 0.03791961  0.20411929 -0.08323022 -0.604376  ]
#[ 0.89689082  0.38273924 -1.09389451  1.72151093]
# [ 1.0089484   0.94152967 -2.28390792 -0.70353106]
# [-0.09475514  0.07616765  0.20620758 -1.51478174]
# [ 0.54183791  1.14011835  0.92304185  0.98882022]
# [ 0.46238628  0.59613739 -0.29512655 -0.15973493]
# [-1.14520718 -0.15275156  1.60839528  0.49325047]
# [ 0.97930329 -0.05914902  0.92119868  2.68343576]]

# use matplot
#data =np.random.normal(size=(10,4))
#lables =['A',"B",'C','D']
#plt.boxplot(data,labels=lables)
#plt.show()

# use seaborn
df = pd.DataFrame(data,columns=lables)
sns.barplot(data=df)
plt.show()