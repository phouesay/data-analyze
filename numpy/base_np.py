import numpy as np

a =np.array([[4,3,2],[2,4,1]])
print(a)
print(np.sort(a))
print(np.sort(a,axis=None))
# 跨行，按列
print(np.sort(a,axis=0))
# 跨列，按行
print(np.sort(a,axis=1))


persontype = np.dtype({
        "names":['name','age','chinese','math','english'],
        'formats':['U32','i','i','i','f']
        })

peoples = np.array([("张飞",32,75,100,90),("关羽",32,75,100,90),("赵云",32,75,100,90),("黄忠",32,75,100,90)],dtype=persontype)

print(peoples)

# 向上取整
print(np.ceil(3.4))