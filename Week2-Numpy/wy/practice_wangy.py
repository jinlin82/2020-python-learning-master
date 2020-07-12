## 03beamer

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

import pandas as pd
# 读取csv格式数据
BSdata=pd.read_csv("../data/BSdata.csv", encoding='utf-8');BSdata
BSdata[6:9]

# 读取Excel格式数据
BSdata=pd.read_excel('../data/DaPy_data.xlsx', 'BSdata');BSdata[-5:]

# 从剪切板上读取
BSdata=pd.read_clipboard()
BSdata[:5]

# pandas数据集的保存
BSdata.to_csv('BSdata1.csv')

# 显示基本信息
BSdata.info() # 显示数据结构
BSdata.head() # 显示数据框前5行
BSdata.tail() # 显示数据框后5行
BSdata.columns # 数据框列名（变量名）
BSdata.index # 数据框行名（样品名）
BSdata.shape # 数据框维度
BSdata.shape[0] # 行数
BSdata.shape[1] # 列数
BSdata.values # 数据框值（数组）

# 选取变量
BSdata.身高 # 取一列数据，BSdata['身高']
BSdata['身高']
BSdata[['身高','体重']]
BSdata.iloc[:,2]
BSdata.iloc[:,2:4]

# 选取样本与变量
BSdata.loc[3]
BSdata.loc[3:5]
BSdata.loc[:3,['身高', '体重']]
BSdata.iloc[:3,:5]

# 条件选取
BSdata[BSdata['身高']>180]
BSdata[(BSdata['身高']>180)&(BSdata['体重']<80)]

# 数据框的运算
BSdata['体重指数']=BSdata['体重']/(BSdata['身高']/100)**2
round(BSdata[:5],2) # 生成新的数据框
BSdata.iloc[:3,:5].T # 数据框转置
pd.concat([BSdata.身高, BSdata.体重], axis=0) # 按行合并
pd.concat([BSdata.身高, BSdata.体重], axis=1) # 按列合并

import numpy as np
A=np.random.randint(0,10,size=(10,5))
B=np.random.randint(0,10,size=(10,5))
A=pd.DataFrame(A)
B=pd.DataFrame(B)
pd.concat([A,B],axis=0)
pd.concat([A,B],axis=1)


## NumPy

# 基本操作
import numpy as np
a=np.arange(4)
b=np.array([2,5,8,9])
a*b

A=np.arange(12).reshape(3,4);A
B=np.arange(13,25).reshape(4,3)
np.dot(A,B)
A.dot(B)
A.sum()
A.sum(axis=0)
A.sum(axis=1)

# 通用函数
A=np.arange(12).reshape(3,4)
np.exp(A)
np.sqrt(A)

# 索引、切片和迭代
x=np.arange(12)**2;x
x[3]
x[2:6]
x[7:]
x[::-1]
x[9:2:-3]

A=np.arange(24).reshape(4,6);A
A[2,3]
A[1:3,2:4]
A[1]
A[:,2:4]
A[...,3]

A=np.arange(24).reshape(4,6);A
for i in A:
    """打印A的各行"""
    print(i)

for i in A.flat:
    """打印A中的每个元素"""
    print(i)

# 改变数组的形状
import numpy as np
a=np.floor(10*np.random.random((3,4)));a #？
a.shape

a.ravel()
a.T
a.reshape(2,6)
a.resize(2,6)
a

# 用一维索引数组进行索引
a=np.arange(12)**2
i=np.array([1,1,3,8,5]) # an array of indices
a[i]

j=np.array([[3,4],[9,7]])
a[j]

# 用二维索引数组进行索引
a=np.arange(12).reshape(3,4);a
i=np.array([[0,1],[1,2]])
j=np.array([[2,1],[3,3]])

a[i]
a[i,j]
a[i,2]
a[:,j]

L=[i,j]
a[L]

# 用布尔数组进行索引
a=np.arange(12).reshape(3,4);a
b=a>4
a[b]
a[b]=0                         

a=np.arange(12).reshape(3,4);a
b1=np.array([False,True,True])
b2=np.array([True,False,True,False])
a[b1,:]
a[b1]
a[:,b2]
a[b1,b2]      

# 用字符串进行索引
x=np.array([('Rex',9,81.0),('Fido',3,27.0)],dtype=[('name','U10'),('age','i4'),('weight','f4')])
x['name']
x[['name','age']]

# ix_()函数
a=np.array([2,3,4,5])
b=np.array([8,5,4])
c=np.array([5,4,6,8,3])
ax,bx,cx = np.ix_(a,b,c)

result=ax+bx * cx
result


## SciPy

# 正态分布
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

mean, var, skew, kurt = stats.norm.stats(moments="mvsk")
x=np.linspace(-3,3,100)
plt.plot(x,stats.norm.pdf(x),label='norm pdf')
plt.plot(x,stats.norm.pdf(x,3,2),label='norm pdf')

### Freeze the distribution and display the frozen pdf
rv=stats.norm(3,2)
rv.ppf(0.5)
rv.pdf(3)
rv.cdf(10)

### Generate random numbers
r = stats.norm.rvs(size=1000)
plt.hist(r,density=True,histtype='stepfilled',alpha=0.2)  ?

# F分布
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

mean, var, skew, kurt = stats.f.stats(3,5,moments="mvsk")
x=np.linspace(0.01,6,100)
plt.plot(x,stats.f.pdf(x,3,5),label='norm pdf')
plt.plot(x,stats.f.pdf(x,3,2),label='norm pdf')

### Freeze the distribution and display the frozen pdf
rv=stats.f(3,2)
rv.ppf(0.5)
rv.pdf(3)
rv.cdf(10)

### Generate random numbers
r = stats.f.rvs(3,2,size=1000)
plt.hist(r,density=True,histtype='stepfilled',alpha=0.2)  

# 二项分布
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

fig,ax=plt.subplots(1, 1)

mean, var, skew, kurt = stats.binom.stats(5,0.4,moments="mvsk")

x=np.arange(stats.binom.ppf(0.01,5,0.4), stats.binom.ppf(0.99,5,0.4))
ax.plot(x,stats.binom.pmf(x,5,0.4),'bo',ms=8,label='binom pdf')
ax.vlines(x,0,stats.binom.pmf(x,5,0.4),colors='b',lw=5,alpha=0.5)

rv=stats.binom(5,0.4)
ax.vlines(x,0,rv.pmf(x),colors='k',linestyles='-',lw=1,label='frozen pmf')
ax.legend(loc='best',frameon=False)

# 多元正态分布
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(0,5,10,endpoint=False)
y=stats.multivariate_normal.pdf(x,mean=2.5,cov=0.5);y

fig1=plt.figure()
ax=fig1.add_subplot(111)
ax.plot(x,y)

x,y=np.mgrid[-1:1:.01,-1:1:.01]
pos=np.dstack((x,y))
rv=stats.multivariate_normal([0.5,-0.2],[2.0,0.3],[0.3,0.5])
fig2=plt.figure()
ax2=fig2.add_subplot(111)
ax2.contourf(x,y,rv.pdf(pos))