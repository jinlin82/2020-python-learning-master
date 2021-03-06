### 对象操作
who
x=10
who
y=20
who
del y
who

### 数值型
n=10
n

print("n=",n)

x=10.234
print(x)

print("x=%10.5f"%x)

### 逻辑型
a=True;a
b=False;b
10>3
10<3
print(3)

### 字符型
s='I <3 coding'
s[0:4]
s[2:4]

len(s)

s[2:11]
s[2:12]
s[2:]

s[-3:]

dir(s)
s.count('i')
s.upper()

s+','+s
s*10

s='IlovePython';s
s[7]
s[2:6]
s+s
s*2

### 缺失值
float('nan')

import math
math.pi

int(math.pi)

### 数据基本类型转换
x=int(3)
float(x)
str(x)
chr(x)
type(x)

[1,4,9,16,25][1:3]
[1,4,9,16,25][3]

### list
list1=[];list1
list1=['Python',786,2.23,'R',70.2]
list1
list1[0]
list1[1:3]
list1[2:]
list1*2
list1+list1[2:4]

X=[1,3,6,4,9];X

sex=['女','男','男','女','男']
sex

weight=[67,66,83,68,70]
weight

x=[1,4,9,16,25]
type(x)
y=(1,4,9,16,25) ### tuple
type(y)

### dictionary
score={'math':85,'stat':90,'com':80}
score['stat']

{}
dict1={'name':'john','code':6734,'dept':'sales'};dict1
dict1=['code']
dict1.keys() #？
dict1.values() #？

dict2={'sex':sex,'weight':weight};dict2

import numpy as np

a=[1,4,9,16]
type(a)
sum(a)

x=np.array(a)
sum(x)
x.var()
x.mean()

### 生成数据
np.arange(10)+1
np.arange(101,201)
np.arange(101,201,step=0.5)

np.linspace(101,200,num=300)

### 一维数组（向量）
np.array([1,2,3,4,5])
np.array([1,2,3,np.nan,5])
np.array(X)
np.arange(9)
np.arange(1,9,0.5)
np.linspace(1,9,5)
np.random.randint(1,9)
np.random.rand(10)
np.random.randn(10)
np.random.randint(1,10,20)
np.random.randint(1,10,(5,3))
np.random.normal(2,3,100)

### 二维数组（矩阵）
np.array([[1,2],[3,4],[5,6]])

A=np.arange(9).reshape((3,3));A

### 数组的操作
A.shape # 数组的维度
np.empty([3,3]) # 空数组
np.zeros([3,3]) # 零数组
np.ones([3,3]) # 1数组
np.eye(3) # 单位阵
np.diag(np.arange(1,5)) # 对角阵
np.diag(np.eye(4))

### pandas
import pandas as pd
pd.Series()

### 根据列表构建序列
X=[1,3,6,4,9]
S1=pd.Series(X);S1
S2=pd.Series(weight);S2
S3=pd.Series(sex);S3

### 系列合并
pd.concat([S2,S3],axis=0)
pd.concat([S2,S3],axis=1)

### 系列切片
S1[2]
S2[1:4]

### 生成数据框
pd.DataFrame()

### 根据列表创建数据框
pd.DataFrame(X)
pd.DataFrame(X,columns=['X'],index=range(5))
pd.DataFrame(weight,columns=['weight'],index=['A','B','C','D','E'])

### 根据字典创建数据框
df1=pd.DataFrame({'S1':S1,'S2':S2,'S3':S3});df1
df2=pd.DataFrame({'sex':sex,'weight':weight},index=X);df2

### 增加数据框列
df2['weight2']=df2.weight**2;df2

### 删除数据框列
del df2['weight2'];df2

### 缺失值处理
df3=pd.DataFrame({'S2':S2,'S3':S3},index=S1);df3
df3.isnull()
df3.isnull().sum()
df3.dropna()

### 数据框排列
df3.sort_index()
df3.sort_values(by='S3')

import pandas as pd
### 读取csv
BSdata=pd.read_csv("../data/BSdata.csv",encoding='GB2312') # ！注意csv文件编码
BSdata[6:9]

### 读取excel
BSdata=pd.read_excel('../data/DaPy_data.xlsx','BSdata')
BSdata[-5:]

### 从剪贴板上读取
BSdata=pd.read_clipboard();
BSdata[:5]

### 数据集的保存
BSdata.to_csv('BSdata1.csv')

### 数据框的操作——显示基本信息
BSdata.info() # 显示数据结构
BSdata.head() # 显示数据框前五行
BSdata.tail() # 显示数据框后五行

BSdata.columns # 数据框列名（变量名）
BSdata.index # 数据框行名（样品名）

# 数据框维度
BSdata.shape
BSdata.shape[0] # 行数
BSdata.shape[1] # 列数

BSdata.values # 数据框值（数组）

# 选取变量
BSdata.身高
BSdata[['身高','体重']]
BSdata.iloc[:,2]
BSdata.iloc[:,2:4]

# 选取样本与变量
BSdata.loc[3]
BSdata.loc[3:5]
BSdata.loc[:3,['身高','体重']]
BSdata.iloc[:3,:5]

# 条件选取
BSdata[BSdata['身高']>180]
BSdata[(BSdata['身高']>180)&(BSdata['体重']<80)]

### 数据框的运算
# 生成新的数据框
BSdata['体重指数']=BSdata['体重']/(BSdata['身高']/100)**2
round(BSdata[:5],2) # 保留两位小数

# 数据框转置
BSdata.iloc[:3,:5].T

# 数据框的合并
pd.concat([BSdata.身高, BSdata.体重],axis=0) # 按行合并
pd.concat([BSdata.身高, BSdata.体重],axis=1) # 按列合并

### Numpy基础
# basic operations
import numpy as np
a=np.arange(4)
b=np.array([2,5,8,9])
a*b

A=np.arange(12).reshape(3,4)
B=np.arange(13,25).reshape(4,3)
np.dot(A,B) # A*B
A.dot(B)
A.sum()
A.sum(axis=0)
A.sum(axis=1)

np.exp(A)
np.sqrt(A)

# indexing, slicing and lterating
x=np.arange(12)**2
x[3]
x[2:6]
x[7:]
x[::-1]
x[9:2:-3]

A=np.arange(24).reshape(4,6)
A
A[2,3] # 取单个元素
A[1:3,2:4] # 取array[行，列]
A[1] # 取一行
A[:,2:4]
A[...,3] # 取一列

A=np.arange(24).reshape(4,6)
for i in A: # 打印A的各行
    print(i)

for i in A.flat: # 打印A中的每个元素
    print (i)

# shape manipulation
a=np.floor(10*np.random.random((3,4)))
a.shape

a.ravel()
a.T
a.reshape(2,6)
a.resize(2,6)

# indexing with arrays of indices-1D
a=np.arange(12)**2 # the first square numbers
i=np.array([1,1,3,8,5]) # an array of indices
a[i] # the elements of a at the position i

j=np.array([[3,4],[9,7]])
a[j]

# indexing with arrays of indices-2D
a=np.arange(12).reshape(3,4)
i=np.array([[0,1],[1,2]])
j=np.array([[2,1],[3,3]])

a[i] # [行，行]
a[i,j] # [行，列]
a[i,2] # [行，第二列]
a[:,j] # 每行取列

L=[i,j]
a[L]

# indexing with boolean arrays ???
a=np.arange(12).reshape(3,4)
b=a>4
a[b]
a[b]=0

a=np.arange(12).reshape(3,4)
b1=np.array([False,True,True])
b2=np.array([True,False,True,False])
a[b1,:]
a[b1]
a[:,b2]
a[b1,b2]

# indexing with strings
x=np.array([('Rex',9,81.0),('Fido',3,27.0)],dtype=[('name','U10'),('age','i4'),('weight','f4')])
x['name']
x[['name','age']]

# the ix_() function
a=np.array([2,3,4,5])
b=np.array([8,5,4])
c=np.array([5,4,6,8,3])
ax,bx,cx=np.ix_(a,b,c)

result=ax+bx*cx
result