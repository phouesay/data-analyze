import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# data prepare
x =[i+1 for i in range(2009,2019)]
y = [5,3,6,20,17,16,19,30,32,35]
# use matplot 
#plt.plot(x,y)
#plt.show()

# use seaborn
df =pd.DataFrame({'x':x,'y':y})
sns.lineplot(x='x',y='y',data= df)
plt.show()