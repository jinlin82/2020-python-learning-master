## 作业内容 ##

## 1. 分别创建一个列表,元组，字典，并提取其子集 

list=[1,2,3,4]   #列表
list[1:3]

tuple=(1,2,3,4)    #元组
tuple[1:3]

dict={'A':11,'B':22,'C':33}   #字典
dict['A']


##  2. 利用 numpy 生成一个一维数组和二维数组

import numpy as np 
x=np.array([1,2,3,4]);x              #一维数组
y=np.array([[1,2],[3,4],[5,6]]);y    #二维数组


##  3. 0-100之间生成等步长的201个数；把列表 [1,4,9] 重复10遍

import numpy as np 
np.linspace(0,100,num=201)

a=[1,4,9]
A=a*10;A


##  4. 随机生成8乘以20的整数array，并得到其转置数组，对其转置数组提取前10行，第2,5,8列数据

import numpy as np 
R=np.random.randint(1,100,size=(8,20));R

RT=R.T;RT     #转置

RT[:10,[1,4,7]]


##  5. 利用pandas把data文件夹中的 数据.xls 导入，要求使用相对路径

import pandas as pd 
data=pd.read_excel('../data/数据.xls',0)


##  6. 显示 数据.xls 数据集的结构，前5条数据，后5条数据，所有变量名，及其维度

data.info()    #数据集结构
data.head()    #前5条数据
data.tail()    #后5条数据
data.columns   #所有变量名
data.shape     #数据集维度


##  7. 对数据集增加一个变量HRsq，其为 HR数据的平方

data['HRsq']=data.HR**2


##  8. 对 数据.xls 数据集 进行子集的提取：

###     1. 删除有缺失值的年份

d1=data.dropna();d1

###     2. 提取逢5，逢0年份的Year, GDP, KR, HRsq，CPI 变量的数据

d2=data.loc[[1,6,11,16,21,26,31,36,41,46,51,56,61],['Year','GDP','KR','HRsq','CPI']];d2

###     3. 提取逢2，逢8年份的Year, GDP, KR, HRsq，CPI 变量的数据

d3=data.loc[[3,9,13,19,23,29,33,39,43,49,53,59],['Year','GDP','KR','HRsq','CPI']];d3

###     4. 对2和3得到的数据集按行进行合并，并按年份进行排序  1111 

import pandas as pd 
d4=pd.concat([d2,d3],axis=0)
D4=d4.sort_values(by='Year');D4

###     5. 提取1978年之后的数据

d5=data[data.Year>1978];d5

###     6. 提取1978年之后且 KR 变量在 1~1.2之间的数据

d6=data[(data.Year>1978)&(data.KR>1)&(data.KR<1.2)];d6


##  9. 保存数据为csv和excel

###     1. 写出第8题中第4问得到的子集到data文件夹

D4.to_csv('../data/data1_lyt.csv')
D4.to_excel('../data/data1_lyt.xlsx')

###     2. 写出第8题中第6问得到的子集到data文件夹

d6.to_csv('../data/data2_lyt.csv')
d6.to_excel('../data/data2_lyt.xlsx')
