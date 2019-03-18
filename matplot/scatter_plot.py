import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

N =1000
x =np.random.randn(N)
y = np.random.randn(N)

# use matplot
#plt.scatter(x,y,marker='X')
#plt.show()

# use seaborn
df = pd.DataFrame({'x':x,'y':y})
sns.jointplot(x="x",y="y",data=df,kind='scatter')
plt.show()