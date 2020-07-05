##1. 分别创建一个列表,元组，字典，并提取其子集
x=[1,2,3,4,5] 
type(x)
x[1:3]

y=(1,2,3,4,5)
type(y)
y[1:3]

z={'a':100,'b':'good','c':'好'}
type(z)
z.keys()
z.values()
z['c']

##2. 利用 numpy 生成一个一维数组和二维数组
import numpy as np
a=[3,6,4,8]
x=np.array(a)
type(x)
x
b=[[1,2],[3,4],[5,6]]
y=np.array(b)
y
type(y)

##3. 0-100之间生成等步长的201个数；把列表 [1,4,9] 重复10遍
import numpy as np
np.linspace(0,100,201)
c='[1,4,9]'
c*10

##4. 随机生成8乘以20的整数array，并得到其转置数组，对其转置数组提取前10行，第2,5,8列数据
import numpy as np
A=np.random.randint(1,10,[8,20])
A.T
A.T[0:10,[1,4,7]]

##5. 利用pandas把data文件夹中的 数据.xls 导入，要求使用相对路径
import pandas as pd
pd.read_excel("../data/数据.xls",0)

##6. 显示 数据.xls 数据集的结构，前5条数据，后5条数据，所有变量名，及其维度
data1=pd.read_excel("../data/数据.xls",0)
data1.info()
data1.head()
data1.tail()
data1.index
data1.columns
data1.shape
data1.shape[0]
data1.shape[1]

##7. 对数据集增加一个变量HRsq，其为 HR数据的平方
data1.HR
data1["HRsq"]=(data1.HR)**2
data1.columns

##8. 对 数据.xls 数据集 进行子集的提取：
###1. 删除有缺失值的年份
data1
data1.isnull()
data1.isnull().sum()
data1.dropna()

###2. 提取逢5，逢0年份的Year, GDP, KR, HRsq，CPI 变量的数据
data1
type(data1)
d2=data1.loc[data1['Year']%5==0,['Year','GDP','KR','HRsq','CPI']]
d2

###3. 提取逢2，逢8年份的Year, GDP, KR, HRsq，CPI 变量的数据
data1
d3=data1.loc[data1['Year']%10==2,['Year','GDP','KR','HRsq','CPI']];d3
da=data1.loc[data1['Year']%10==8,['Year','GDP','KR','HRsq','CPI']];da
db=pd.concat([d3,da],axis=0);db
dc=db.sort_values(by='Year');dc

###4. 对2和3得到的数据集按行进行合并，并按年份进行排序  1111 
d4=pd.concat([d2,d3],axis=0);d4
d5=d4.sort_values(by='Year');d5

###5. 提取1978年之后的数据
data1
d6=data1[data1['Year']>1978];d6

###6. 提取1978年之后且 KR 变量在 1~1.2之间的数据
d7=data1[(data1['Year']>1978)&(data1['KR']>1)&(data1['KR']<1.2)]

##9. 保存数据为csv和excel
###1. 写出第8题中第4问得到的子集到data文件夹
d5
d5.to_csv('../data/xxy1.csv')
d5.to_excel('../data/xxy1.xlsx')

###2. 写出第8题中第6问得到的子集到data文件夹
d7
d7.to_csv('../data/xxy2.csv')
d7.to_excel('../data/xxy2.xlsx')



