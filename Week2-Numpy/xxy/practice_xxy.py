##03——beamer
who
x=10.12
who
del x
who

n=10
n
print("n=",n)
x=10.1234
print(x)
print("x=%10.6f"%x)

a=True;a
b=False;b
10>3
10<3
print(3)
1==2
2>3

s='I love Python';s
s[7]
s[2:6]
s[1:7]
s+s
s*3

float('nan')
list1=[];list1
list1=['python',786,2.23,'R',70.2]
list1
type(list1)
list1[0]
list1[1:3]
list1[2:]
list1*2
list1+list1[2:4]

x=[1,3,4,6,9];x
sex=['女','男','男','女','男']
sex
weight=[67,66,83,68,70]
weight

x=(1,2,3);x
type(x)

dict1={'name':'john','code':6734,'dept':'sales'};dict1
dict1['code']
dict1.keys()
dict1.values()
dict2={'sex':sex,'weight':weight};dict2

import numpy as np
np.array([1,2,3,4,5])
np.array([1,2,3,np.nan,5])
np.array(x)
np.arange(9)
np.arange(1,9,0.5)
np.linspace(1,9,50)
np.random.randint(1,9,[3,4])
np.random.rand(2,3)
np.random.rand(10)
np.random.randn(10)
np.random.randn(1000).mean()
np.random.randn(1000).var()

np.array([[1,2],[3,4],[5,6]])
A=np.arange(9).reshape((3,3));A
A.shape
np.empty([2,3])
np.zeros([3,5])
np.ones([3,4])
np.eye(4)

import pandas as pd 
pd.Series()
x=[1,2,3,4,5]
S1=pd.Series(x);S1
S2=pd.Series(weight);S2
S3=pd.Series(sex);S3
pd.concat([S2,S3],axis=0)
pd.concat([S2,S3],axis=1)
type(S1)
S1[2]
S2[1:4]

pd.DataFrame(x)
pd.DataFrame(x,columns=['x'],index=range(5))
pd.DataFrame(weight,columns=['weight'],index=['A','B','C','D','E'])

df1=pd.DataFrame({'S1':S1,'S2':S2,'S3':S3});df1
df2=pd.DataFrame({'sex':sex,'weight':weight},index=x);df2
type(df1)

df2['weight']=df2.weight**2;df2

del df2['weight'];df2
df3=pd.DataFrame({'S2':S2,'S3':S3},index=S1);df3
df3.isnull()
df3.isnull().sum()
df3.dropna()
df3.sort_index()
df3.sort_values(by='S2')

import pandas as pd
BSdata=pd.read_csv("./data/BSdata.csv",encoding='utf-8')
BSdata[1:3]
BSdata=pd.read_excel("./data/DaPy_data.xlsx",0)
BSdata[-3:]

BSdata.to_csv('BSdata1.csv')
BSdata.to_csv('./data/BSdata2.csv')
BSdata.info()
BSdata.head()
BSdata.tail()
BSdata.columns ##数据框行名（变量名）
BSdata.index ##数据框列名（样品名）
BSdata.shape
BSdata.shape[0]
BSdata.shape[1]
BSdata.values
BSdata.身高
BSdata[['身高','体重']]
BSdata.iloc[3:5]
BSdata.iloc[3]
BSdata.iloc[:,2:4]
BSdata.loc[3]
BSdata.loc[3:5]
BSdata.loc[:3,['身高','体重']]
BSdata.iloc[:3,:5]
BSdata[BSdata['身高']>180]
BSdata[(BSdata['身高']>180)&(BSdata['体重']<80)]

BSdata['体重指数']=BSdata['体重']/(BSdata['身高']/100)**2
round(BSdata[:5],4) ##保留小数点位数
BSdata.iloc[:3,:5].T

pd.concat([BSdata.身高,BSdata.体重],axis=0)
pd.concat([BSdata.身高,BSdata.体重],axis=1)

## numpy
import numpy as np
x=np.array([1,4,9,16])
type(x)
type([1,4,9,16])

a=[1,4,9,16]
type(a)
sum(a)
dir(a)
a.var()
mean(a)

x=np.array(a)
x
np.var(x)
type(x)
sum(x)
dir(x)
x.var()
x.mean()

### 生成数据
np.arange(10)
np.arange(10)+1

np.arange(101,200)
## arange 包括100 不包括200
np.arange(101,201)

np.arange(101,201,0.5)
m=np.arange(51,151,3) 
## 被3整除的数，第一个数，然后步长
m
len(m)

np.linspace(101,200,num=300)
## linspace 101-200 包括101 200

np.linspace(1,100,num=100)
np.linspace(1,1,num=100)

np.random.seed(10)
np.random.randint(1,10,size=20)
np.random.randint(1,10,size=[6,4])
## size=数字 生成一维 size=矩阵 生成二维
np.random.rand(100)
np.random.randn(100).var()
np.random.randn(100).mean()
np.random.randn(100).var()
np.random.normal(2,4,100).var()
np.random.normal(2,4,100).mean()

np.array([[1,2],[3,4],[5,6]])
np.random.seed(1234)
np.random.randint(1,10,size=(5,4))
A=np.arange(12).reshape((4,3))
A.reshape((6,2),order='C')

A.shape
np.empty([3,3])
np.zeros((4,3))
np.ones((2,2))
np.eye(4)
np.diag(np.arange(1,5)) 
np.diag(np.eye(5))
np.diag(np.arange(3,6))

import pandas as pd
X=[1,3,6,4,9]
S1=pd.Series(X);S1
S1[2]
x=[1,2,3,4]
y=[8,10,20,16,5]
type(x)
dir(x)
sum(x)
x.var()
type(y)
x1=np.array(x)
y1=np.array(y)
type(x1)
x1.var()
type(y1)
x2=pd.Series(x)
y2=pd.Series(y)
type(x2)
type(y2)
x2.var()
x2.plot()
dir(x1)
dir(x2)
pd.concat((x2,y2),axis=0)
pd.concat((x2,y2),axis=1)
y2[2]
y2[0:4]
y2[2:3]

x=[[1,2],[3,4],[5,6]]
type(x)
x1=np.array([[1,2],[3,4],[5,6]])
type(x1)
pd.DataFrame(x)

x2=pd.DataFrame(x1)
type(x2)
dir(x2)

y1={'name':['zhang','wang','li'],'stat':[80,75,93],'math':[90,84,87],'comp':[85,92,96]}
score=pd.DataFrame(y1)
type(y1)
y1.keys()
y1.values()

score['English']=[82,85,91]
score

score['sports']=[79,np.nan,93]
score
del score['comp']
score
score.isnull()
score.isnull().sum()
score.mean()
score.dropna()
dir(score)

score
score.sort_index()
score.sort_index(axis=1)

score.sort_values(by='stat')
score.sort_values(by=['stat','math'])
score.sort_values(by='stat',ascending=False)

import pandas as pd
BSdata=pd.read_csv("./data/BSdata.csv",encoding="utf-8")
type(BSdata)
BSdata[1:5]
dir(BSdata)

BSdata2=pd.read_excel("./data/DaPy_data.xlsx",1)
BSdata2[0:3]

# dat=pd.read_clipboard()

del BSdata["学号"]
BSdata["性别"]

# 函数
BSdata.to_csv("./data/new.csv")
BSdata.to_excel("./data/new.xlsx")

BSdata=pd.read_excel("./data/DaPy_data.xlsx",0)
type(BSdata)
dir(BSdata)
、
# 方法
BSdata.info()
# 显示数据结构
BSdata.head()
BSdata.tail()
BSdata[:10]
BSdata.head(10)
BSdata.tail(4)

# alt+shift+上/下箭头 可复制特指某一行代码
# alt+上/下箭头对该行代码进行移动
# ctrl+/ 将该行转化为注释

BSdata.sort_values('支出',ascending=False)
# 默认从下到大排序 false则为从大到小

BSdata.sort_index(axis=1,ascending=False)
# axis=0 按行排序 axis=1 按列排序 中文下未知，英文下按首字母顺序排序

# 属性
BSdata.index
BSdata.columns
BSdata.values
BSdata.keys

BSdata.shape
type(BSdata.shape)
BSdata.shape[0]
BSdata.shape[1]

import numpy as np
A=np.random.randint(0,10,[10,5])
A.shape
A.reshape(25,2)
# shape属性（固有） reshape 方法（对其进行操作）

BSdata["体重"]
BSdata["体重指数"]=BSdata.体重/(BSdata.身高/100)**2
BSdata.columns

round(BSdata[:10],3)
# 后面数字表示保留的小数位数

BSdata.T
# 转置为属性 不是对对象进行操作 不需要（）
BSdata
dir(BSdata)
BSdata.reshape()

B=np.random.randint(0,10,[10,5])
B

A=pd.DataFrame(A)
B=pd.DataFrame(B)
pd.concat([A,B],axis=0)
# 按行合并
pd.concat([A,B],axis=1)
# 按列合并

