#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:32:05 2019

@author: chance
"""
# -*- coding: utf-8 -*-
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties



# data prepare
labels =np.array([u' 推进 ','KDA',u' 生存 ',u' 团战 ',u' 发育 ',u' 输出 '])
stats =[83,61,95,67,76,88]
# draw
angles =np.linspace(0,2*np.pi,len(labels),endpoint=False)
stats =np.concatenate((stats,[stats[0]]))
angles = np.concatenate((angles,[angles[0]]))
# use matplot
fig = plt.figure()
ax = fig.add_subplot(111,polar=True)
ax.plot(angles,stats,'o-',linewidth =2)
ax.fill(angles,stats,alpha =0.25)

#font = FontProperties(fname="",size=14)
#ax.set_thetagrids(angles*180/np.pi,labels,FontProperties= font)
ax.set_thetagrids(angles*180/np.pi,labels)
plt.show()