## 作业内容 ##

import pandas as pd
import numpy as np
# 1. 使用循环数据集，
   ##1. 用 for 循环读入所有csv文件
import os
files=os.listdir("./data")
data=pd.DataFrame()
for i in files:
   temp=pd.read_csv('./data/'+i,encoding='gb2312')
   ##2. 把所有读入的csv数据分别转换成一列向量的形式
   temp1=temp.iloc[:,1:].values.T.reshape(-1)
   data[i[:-4]]=temp1
   
   ##3. 创建时间和地区的面板数据的表头
year=np.arange(2002,2012)
dis=temp.district

District,Year=np.meshgrid(dis,year)
Year=pd.Series(Year.reshape(-1)) 
District=pd.Series(District.reshape(-1)) #也可以转化为list*10

##4. 把表头和41个列向量合并成一个数据框
data=pd.concat((Year,District,data),axis=1);data

   ##5. 对数据框的变量名进行修改为time, dis和 41 个 csv文件的名字
data.columns.values[:2]=['time','dis']

   ##6. 把最后得到的数据框写出为csv文件
data.to_csv('./data.csv')

# 2. 自定义函数
   ##1. 给出一个 list， 求其平均差 $MAD=\frac{1}{n}\sum_{i}^{n}|x_{i}-\bar x_{i}|$
x=[1,2,3,4,5,6,7,8,9,10]
def mad(x):
   mean=np.mean(x)
   a=list()
   for i in x:
      a.append(np.abs(i-mean))
   n=len(x)
   m=sum(a)
   return m/n

mad(x)

   ##2. 编写一个函数opposite，把向量倒置，对某一向量使用该函数
a=np.arange(10);a
def opposite(x):
   y=x[::-1]
   return y
opposite(a)

   ##3. 编写一个函数shift，把向量元素右移 k 个位置，对某一向量使用该函数
a=np.arange(10);a
def shift(x,k):
   if k>len(x):
      k=k%len(x)
   y=list(x[len(x)-k:])
   z=list(x[:len(x)-k])
   return y+z
shift(a,3)
shift(a,11)

   ##4. 生成一个20行10列的矩阵，把矩阵的每一列倒置，把矩阵的每一行元素向右3个位置
a=np.arange(200).reshape((20,10));a
b=opposite(a);b
c=list()
for i in b:
   i=shift(i,3)
   c.append(i)
c
##法二##
np.apply_along_axis(lambda y:shift(y.tolist(),3),1,np.apply_along_axis(opposite,0,a))


   ##5. 编写一个函数 fibonacci ，给定一个正整数x, 生成小于x的所有斐波那契数列元素， 求x=10000000时具体数列.
def fibo(x):
   x1=x2=1
   x3=[1,1]
   while x3[-1]<=x:
      x1=x2
      x2=x3[-1]
      x3.append(x1+x2)
   return x3[:-1]
fibo(10000000)

## 法二
def fib(n):
   a,b=0,1
   while a<n:
      print(a)
      a,b=b,a+b
fib(10000000)


# 3. 自定义一个正方形的类：
   ##1. 给出其位置和边长属性
class square1:
    """一个二维平面中的正方形"""
    pos=(0,0)
    s=1

   ##2. 更改其 __init__ 方法，位置和边长属性默认为(0,0)和1
class square1:
    """一个二维平面中的正方形"""
    def __init__(self,t=(0,0),x=1):
        self.pos=t
        self.s=x

   ##3. 更改其 __str__ 方法
class square1:
    """一个二维平面中的正方形"""
    def __init__(self,t=(0,0),x=1):
        self.pos=t
        self.s=x
    def __str__(self):
        print('正方形：', '位置为：', self.pos, '边长为：',self.s)
   ##4. 定义一个求其面积的方法
class square1:
   """一个二维平面中的正方形"""
   def __init__(self,t=(0,0),x=1):
      self.pos=t
      self.s=x
   def __str__(self):
      print('正方形：', '位置为：', self.pos, '边长为：',self.s)
   def s_area(x):
      return x.s**2
   ##5. 定义一个两个正方形距离的方法
class square1:
   """一个二维平面中的正方形"""
   def __init__(self,t=(0,0),x=1):
      self.pos=t
      self.s=x
   def __str__(self):
      print('正方形：', '位置为：', self.pos, '边长为：',self.s)
   def s_area(x):
      return x.s**2
   def distance(self,other):
      return ((self.pos[1]-other.pos[1])**2+(self.pos[0]-other.pos[0])**2)**0.5