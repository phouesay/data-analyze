import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

# data prepare
s =np.random.randn(100)
# use matplot
plt.hist(s)
plt.show()


# use seaborn
sns.distplot(s,kde=False)
plt.show()

# kde change True
sns.distplot(s,kde =True)
plt.show()