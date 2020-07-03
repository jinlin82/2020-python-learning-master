# 作业内容

## 1. 分别创建一个列表，元组，字典，并提取其子集 

list=[1,2,3,4,5,6,7,8,9,10];list
list[3:7]

tuple=(1,4,9,16,25);tuple
tuple[:3]

dic={'name':'lmk','sex':'女','birth':980610};dic
dic['birth']

## 2. 利用 numpy 生成一个一维数组和二维数组

import numpy as np
np.array([1,2,3,4,5])
np.array([[1,3],[2,4],[3,5]])

## 3. 0-100之间生成等步长的201个数；把列表 [1,4,9] 重复10遍

np.linspace(0,100,num=201)

list1=[1,4,9];list1
list1*10

## 4. 随机生成8乘以20的整数array，并得到其转置数组，对其转置数组提取前10行，第2,5,8列数据

A=np.ndarray((8,20),dtype=int);A
A.T
A.T[:10,[1,4,7]]

## 5. 利用pandas把data文件夹中的 数据.xls 导入，要求使用相对路径

import pandas as pd
data=pd.read_excel('../data/数据.xls','Sheet1')

## 6. 显示 数据.xls 数据集的结构，前5条数据，后5条数据，所有变量名，及其维度

data.info()
data.head()
data.tail()
data.columns
data.shape

## 7. 对数据集增加一个变量HRsq，其为HR数据的平方

data['HRsq']=data['HR']**2

## 8. 对 数据.xls 数据集 进行子集的提取：

### 1. 删除有缺失值的年份

data1=data.dropna();data1

### 2. 提取逢5，逢0年份的Year, GDP, KR, HRsq, CPI变量的数据

data2=data1.loc[[6,11,16,21,26,31,36,41,46,51,56,61],['Year','GDP','KR','HRsq','CPI']];data2

### 3. 提取逢2，逢8年份的Year, GDP, KR, HRsq, CPI变量的数据

data3=data1.loc[[9,13,19,23,29,33,39,43,49,53,59],['Year','GDP','KR','HRsq','CPI']];data3

### 4. 对2和3得到的数据集按行进行合并，并按年份进行排序

d4=pd.concat([data2,data3],axis=0)
data4=d4.sort_values(by='Year');data4

### 5. 提取1978年之后的数据

data5=data1[data1['Year']>1978];data5

### 6. 提取1978年之后且KR变量在1~1.2之间的数据

data6=data5[(data5['KR']>1)&(data5['KR']<1.2)];data6

## 9. 保存数据为csv和excel

### 1. 写出第8题中第4问得到的子集到data文件夹

data4.to_csv('../data/data4.csv')
data4.to_excel('../data/data4.xlsx')

### 2. 写出第8题中第6问得到的子集到data文件夹

data6.to_csv('../data/data6.csv')
data6.to_excel('../data/data6.xlsx')