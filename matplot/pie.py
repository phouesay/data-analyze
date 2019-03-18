#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:23:59 2019

@author: chance
"""

import matplotlib.pyplot as plt

nums =[25,37,33,37,6]
labels =["High-school","Bachelor","Master","Ph.d","Others"]

# use matplot
plt.pie(x=nums,labels=labels)
plt.show()