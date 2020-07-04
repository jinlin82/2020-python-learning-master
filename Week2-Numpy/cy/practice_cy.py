#03 PDF

##object
who
x = 10.12
who
del x
who
### who/del “内置函数名”

##numeric
n = 10
print("n=",n)
x = 10.234
print(x)
print("x=%10.5f"%x)
y = 20.05
print("y=%10.5f"%y)
###Question?10.5f???

##logic
a = True;a
b = False;b
10>3
10<3
print(3)

##character
s = 'I love Python'
s[7]
s[2:6]
s+s ##"+"字符串连接运算符
s*2 ##"*" do it again

##nan: not available number
float('nan')

##change type
int('123') # turn x in integer 不写base默认10进制
int('123',8)
float(x) # turn x into floating point
str(x) #...into character string
y = 8
chr(y) #turn integer into character
type(chr(y))

##list
list1 = [];list1
list1 = ['Python',786,2.23,'R',70.2]
list1
list1[0]
list1[2]
list1[1:2]###??????
list[2:]
list1*2
list1+list1[2:4]

##
X = [1,3,6,4,9];X
sex = ['女','男','男','女','男']
sex
weight = [67,66,83,68,70];
weight

##tuple(元组)

##dictionary
## dict = {key1:value1,key2:value2}
{}
dict1 = {'name':'jojn','code':6734,'dept':'sales'};dict1
dict1['code']
dict1.keys()
dict1.values()
dict2 = {'sex':sex,'weight':weight};dict2

#数值分析库 numpy

#import numpy as np
#一组数组（向量）
import numpy as np
np.array([1,2,3,4,5])
np.array([1,2,3,np.nan,5])
np.array(X)
np.arange(9)
np.arange(1,9,0.5)
np.linspace(1,9,5)
np.random.randint(1,9) #int
np.random.rand(10) #float
np.random.randn(10) # 有负数
#二维数组（矩阵）
np.array([[1,2],[3,4],[5,6]])
A = np.arange(9).reshape((3,3));A

#数组的操作
A.shape
np.empty([3,3])
np.zeros((3,3))
np.ones((3,3))
np.eye(3)

#数据分析库PANDAS
import pandas as pd

#生成序列
pd.Series()
#根据列表构建序列
X = [1,3,6,4,9]
S1 = pd.Series(X);S1
S2 = pd.Series(weight);S2
S3 = pd.Series(sex);S3
#合并
pd.concat([S2,S3],axis = 0)
pd.concat([S2,S3],axis = 1)
#切片
S1[2]
S3[1:4]
#DataFrame
#生成数据框
pd.DataFrame()
#根据列表创建数据框
pd.DataFrame(X)
pd.DataFrame(X,columns = ['X'],index = range(5))
pd.DataFrame(weight,columns = ['weight'],index = ['A','B','C','D','E'])
#根据字典创建数据框
df1 = pd.DataFrame({'S1':S1,'S2':S2,'S3':S3});df1
df2 = pd.DataFrame({'sex':sex,'weight':weight},index = X);df2
#增加数据框
df2['weight2'] = df2.weight**2;df2
#删除数据框列
del df2['weight2'];df2
#ran 处理
df3 = pd.DataFrame({'S2':S2,'S3':S3},index = S1);df3
df3.isnull()
df3.isnull().sum()
df3.dropna()
#sort
df3.sort_index()
df3.sort_values(by = 'S3')

##pandas读数据
BSdata = pd.read_csv("../data/BSdata.csv",encoding = 'utf-8')
BSdata[6:9]
BSdata = pd.read_excel('../data/DaPy_data.xlsx','BSdata');BSdata[-5:]

BSdata = pd.read_clipboard();
BSdata[:5]

#save data
BSdata.to_csv('BSdata1.csv')

BSdata.info() #show structure
BSdata.head()
BSdata.tail()

#show information
BSdata.columns #col name
BSdata.index #sample name
BSdata.shape #行列数
BSdata.shape[0] #行数
BSdata.shape[1] #列数
BSdata.values #数据框值

#选取变量下表法
BSdata.身高
BSdata['身高']
BSdata["身高"]

BSdata[['身高'，'体重']]
BSdata.iloc[:,2]
BSdata.iloc[:,2:4]

#选取样本和变量
BSdata.loc[3]#名字为3的样本信息
BSdata.iloc[3]
BSdata.loc[:3,['身高','体重']]
BSdata.iloc[:3,:5]

#条件选取
BSdata[BSdata['身高']>180]
BSdata[(BSdata['身高']>180)&(BSdata['体重']<80)]

#generate
BSdata['体重指数'] = BSdata['体重']/(BSdata['身高']/100)**2
round(BSdata[:5],2)
#转置
BSdata.iloc[:3,:5].T

##
pd.concat([BSdata.身高,BSdata.体重],axis = 0) #按行合并
pd.concat([BSdata.身高,BSdata.体重],axis = 1) #按列合并


#numpy
import numpy as np
a = np.arange(4)
b = np.array([2,5,8,9])
a*b

A = np.arange(12).reshape(3,4)
B = np.arange(13,25).reshape(4,3)
np.dot(A,B) #矩阵相乘，下同
A.dot(B)
A.sum()
A.sum(axis = 0) #列和
A.sum(axis = 1) #行和

#function
A = np.arange(12).reshape(3,4)
np.exp(A)
np.sqrt(A) #开根号

x = np.arange(12)**2
x[3]
x[2:6]
x[7:]
x[::-1]
x[9:2:-3]

A = np.arange(24).reshape(4,6)
A[2,3]
A[1:3,2:4]
A[1]
A[:,2:4]
A[...,3]
A[:,3]

import numpy as np
A = np.arange(24).reshape(4,6)
for i in A:
    print(i) #输出为一个矩阵
for i in A.flat:
    print(i) #输出为一列

#changing the shape of an array

import numpy as np
a = np.floor(10*np.random.random((3,4)))
a.shape

a.ravel() #把它变成一维
a.T
a.reshape(2,6)
a.resize(2,6)

#index

a = np.arange(12)**2 #the first 12 square numbers
i = np.array([1,1,3,8,5]) # an array of indices
a[i] # the elements of a at the positions i 

j = np.array([[3,4],[9,7]])
a[j]

a = np.arange(12).reshape(3,4)
i = np.array([[0,1],[1,2]])
j = np.array([[2,1],[3,3]])

a[i] #第0、1行；第1、2行
a[i,j] #(0,2)(1,1)(1,3)(2,3)
a[i,2] #(0,2)(1,2)(1,2)(2,2)
a[:,j] #每一行来选择列

L = [i,j]
a[L]

a = np.arange(12).reshape(3,4)
b = a>4
a[b]
a[b] = 0

a = np.arange(12).reshape(3,4)
b1 = np.array([False,True,True])
b2 = np.array([True,False,True,False])
a[b1,:]
a[b1]
a[:,b2]
a[b1,b2] #？？？

x = np.array([('Rex',9,81.0),('Fido',3,27.0)],dtype = [('name','U10'),('age','i4'),('weight','f4')])
x['name']
x[['name','age']]

#function ???
a = np.array([2,3,4,5])
b = np.array([8,5,4])
c = np.array([5,4,6,8,3])
ax,bx,cx = np.ix_(a,b,c)

result = ax+bx*cx
result