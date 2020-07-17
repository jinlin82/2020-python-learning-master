# 作业内容

## 1. 使用循环数据集

### 1. 用for循环读入所有csv文件
### 2. 把所有读入的csv数据分别转换成一列向量的形式

import pandas as pd
import os
path='../data/'
files=os.listdir(path)
df=pd.DataFrame()
for i in files:
    file=pd.read_csv('../data/'+i,encoding='gb2312')
    file1=file.iloc[:,1:].values.T.reshape(180)
    df[i[:-4]]=file1
df

### 3. 创建时间和地区的面板数据的表头

import numpy as np
year=np.repeat(np.arange(2002,2012),18)
district=list(file.district)*10
df['year']=year
df['district']=district

### 4. 把表头和41个列向量合并成一个数据框

df1=pd.concat([df['year'],df['district'],df],axis=1)
df2=df1.iloc[:,:-2];df2

### 5. 对数据框的变量名进行修改为time, dis和41个csv文件的名字

df3=df2.rename(columns={'year':'time'})
df4=df3.rename(columns={'district':'dis'});df4

### 6. 把最后得到的数据框写出为csv文件

df4.to_csv('dataframe.csv')

## 2. 自定义函数

### 1. 给出一个list，求其平均差 $MAD=\frac{1}{n}\sum_{i}^{n}|x_{i}-\bar x_{i}|$

def MAD(lst):
    n=len(lst)
    mean=sum(lst)/n
    s=0
    for i in range(n):
        s=s+abs(lst[i]-mean)
    return s/n

list=[0,1,2,3,4]
MAD(list)

### 2. 编写一个函数opposite，把向量倒置，对某一向量使用该函数

def opposite(a):
    v=np.flipud(a)
    return v

a=np.arange(6)
v=opposite(a);v

### 3. 编写一个函数shift，把向量元素右移k个位置，对某一向量使用该函数

def shift(a,k):
    n=len(a)
    v1=np.array(a[n-k:])
    v2=np.array(a[:n-k])
    v=np.append(v1,v2)
    return v

a=np.array([35,74,26,95,18,51])
v=shift(a,2);v

### 4. 生成一个20行10列的矩阵，把矩阵的每一列倒置，把矩阵的每一行元素向右3个位置

x=np.random.random((20,10))*10;x
x1=opposite(x)
a=[]
for i in x1:
    i=shift(i,3)
    a.append(i)
x2=np.array(a);x2   # 有没有更好的方法？

### 5. 编写一个函数fibonacci，给定一个正整数x, 生成小于x的所有斐波那契数列元素，求x=10000000时具体数列.

def fibonacci(x):
    a=[1,1]
    b=2
    while b<x:
        a.append(b)
        b=a[-2]+a[-1]
    return a

x=10000000
a=fibonacci(x);a

## 3. 自定义一个正方形的类

### 1. 给出其位置和边长属性

class square:
    """一个二维平面中的正方形"""
    def __init__(self,t,x):
        self.pos=t
        self.l=x

### 2. 更改其 __init__ 方法，位置和边长属性默认为(0,0)和1
### 3. 更改其 __str__ 方法
### 4. 定义一个求其面积的方法
### 5. 定义一个两个正方形距离的方法

class square:
    """一个二维平面中的正方形"""
    def __init__(self,t=(0,0),x=1):   # 更改__init__方法
        self.pos=t
        self.l=x
    def __str__(self):   # 更改__str__方法
        return print('正方形：','位置为：',self.pos,'边长为：',self.l)
    def area(self):   # 定义面积
        return self.l**2
    def dis2(self,other):   # 定义距离
        return((self.pos[0]-other.pos[0])**2+(self.pos[1]-other.pos[1])**2)**0.5

a=square()
a.pos
a.l
a.__str__()
a.area()
b=square((7,6),3)
a.dis2(b)