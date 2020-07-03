# 对象操作
who
x=10.12
who
del x
who

y=20
who
del y
who

# 数值型
n=10
n
print("n=",n)
x=10.234
print(x)
print("x=%10.5f"%x)

# 逻辑型
a=True;a
b=False;b
10>3
10<3
print(3)

# 字符型
s='IlovePython';s
len(s)
s[7]
s[2:6]
s+s
s*2
s[-6:]
dir(s)
s.count('o')
s.upper()

# 缺失值
float('nan')

import math
math.pi
type(math.pi)
int(math.pi)

x=int(3)
float(x)
str(x)
chr(x)

import numpy as np
y=np.long(3)
y
x
type(y)

# 列表
list1=[];list1
list1=['Python',786,2.23,'R',70.2]
list1
list1[0]
list1[1:3]
list1[2:]
list1*2
list1+list1[2:4]

[1,4,9,16,25][1:3]
[1,4,9,16,25][4]

x=[1,4,9,16,25]
type(x)
dir(x)

x.append(36)
x

y=(1,4,9,16,25)
type(y)
dir(y)

x=[1,3,6,4,9];x
sex=['女','男','男','女','男']
sex
weight=[67,66,83,68,70]
weight

# 字典
{}
dict1={'name':'john','code':6734,'dept':'sales'};dict1
dict1['code']
dict1.keys()
dict1.values()
dict2={'sex':sex,'weight':weight};dict2

score={'math':85, 'stat':90, 'com':80}
score['stat']


# 一维数组（向量）
import numpy as np
np.array([1,2,3,4,5])
np.array([1,2,3,np.nan,5])
np.array(x)
np.arange(9)
np.arange(10)+1
np.arange(101,201)
np.arange(1,9,0.5)
np.linspace(1,9,5)
np.linspace(1,1,100)
np.random.randint(1,9)

np.random.seed(10)
np.random.randint(1, 10, 20)
np.random.randint(1, 10, size=(5,3))
np.random.rand(10)
np.random.randn(100).var()

np.random.normal(2,3,100).var()

x=np.array([1,4,9,16])
type(x)
type([1,4,9,16])
sum(x)
dir(x)
x.var()
x.mean()

# 二维数组
np.array([[1,2],[3,4],[5,6]])
np.random.seed(1234)
np.random.randint(1,10,size=(5,4))
A=np.arange(12).reshape((4,3));A
A.reshape((2,6))

# 数组的操作
A.shape
np.empty((3,3))
np.zeros((3,3))
np.ones((3,3))
np.eye(3)
np.diag(np.arange(1,5))
np.diag(np.eye(4))
import pandas as pd

# 生成序列
pd.Series()

# 根据列表构建序列
x=[1,2,3,4]
y=[8,10,20,16,5]

x1=np.array(x)
y1=np.array(y)
type(x1)
type(y1)

x2=pd.Series(x)
y2=pd.Series(y)
type(x2)
type(y2)


X=[1,3,6,4,9]
S1=pd.Series(X);S1

S2=pd.Series(weight);S2

S3=pd.Series(sex);S3

# 系列合并
pd.concat([S2,S3],axis=0)
pd.concat([S2,S3],axis=1)

# 系列切片
S1[2]

S3[1:4]

# 生成数据框
pd.DataFrame()

# 根据列表创建数据框
pd.DataFrame(X)
pd.DataFrame(X,columns=['X'],index=range(5))
pd.DataFrame(weight,columns=['weight'],index=['A','B','C','D','E'])

# 根据字典创建数据框
df1 = pd.DataFrame({'S1':S1, 'S2':S2, 'S3':S3});df1
df2 = pd.DataFrame({'sex':sex, 'weight':weight}, index=X);df2

y1 = {'name':['zhang','wang','zhao'],'math':[80,70,90],'stat':[90,94,88],'com':[85,92,83]}
score = pd.DataFrame(y1)

# 增加数据框列
df2['weight2'] = df2.weight**2;df2

# 删除数据框列
del df2['weight2'];df2

# 缺失值处理
df3 = pd.DataFrame({'S2':S2, 'S3':S3}, index=S1);df3
df3.isnull()
df3.isnull().sum()
df3.dropna()

# 数据框排序
df3.sort_index()
df3.sort_index(axis=1)
df3.sort_values(by='S3')
score.sort_values(by=['stat','math'])
score.sort_values(by='stat',ascending=False)