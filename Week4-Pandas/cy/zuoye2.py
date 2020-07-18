#1. 使用循环数据集

##1. 用 for 循环读入所有csv文件
##2. 把所有读入的csv数据分别转换成一列向量的形式

import numpy as np
import pandas as pd 
import os 
path = './data'
csvfiles = os.listdir(path)

dat = pd.DataFrame() #建立一个空的数据框用于放后面的数据
for i in csvfiles:
    temp = pd.read_csv(path+'/'+i,encoding='gb2312')
    temp1 = temp.iloc[:,1:].values.T.reshape(-1)
    dat[i[:-4]] = temp1

#temp.iloc[:,1:].shape

##3. 创建时间和地区的面板数据的表头
##4. 把表头和41个列向量合并成一个数据框
##5. 对数据框的变量名进行修改为time, dis和 41 个 csv文件的名字

year = np.repeat((np.arange(2002,2012)),18)
Year = pd.DataFrame(list(year),columns=['Year'])#数据只有一列时,columns必须加上中括号
District = pd.DataFrame(np.tile(temp.iloc[:,0],10),columns=['District'])
dat = pd.concat([Year,District,dat],axis=1)# 1为按列合并

##6. 把最后得到的数据框写出为csv文件

dat.to_csv('./dat.csv',encoding='gb2312')

#2. 自定义函数
##1. 给出一个 list， 求其平均差 $MAD=\frac{1}{n}\sum_{i}^{n}|x_{i}-\bar x_{i}|$

def MAD(x):
    n = len(x)
    mean = np.mean(x)
    a = list()
    for k in x:
        a.append(np.abs(k-mean))
    s = sum(a)
    return s/n
x = [1,2,3,4,5]
MAD(x)

##2. 编写一个函数opposite，把向量倒置，对某一向量使用该函数

def opposite(x):
    y = x[::-1] #间隔为-1,即倒着取
    return y
x = np.arange(5);x
opposite(x)

##3. 编写一个函数shift，把向量元素右移 k 个位置，对某一向量使用该函数

def shift(x,k):
   a=list(x[len(x)-k:])
   b=list(x[:len(x)-k])
   return a+b

x = np.arange(5)
shift(x,2)


##4. 生成一个20行10列的矩阵，把矩阵的每一列倒置，把矩阵的每一行元素向右3个位置

a=np.arange(200).reshape((20,10));a
b=opposite(a);b
c=list()
for i in b:
   i=shift(i,3)
   c.append(i)
c

x = np.arange(200).reshape((20,10));x
y = opposite(x);y #将矩阵每一列倒置 每一行怎么做呢？
z = list()
for k in y:
    k = shift(k,3)
    z.append(k)
z

##5. 编写一个函数 fibonacci ，给定一个正整数x, 生成小于x的所有斐波那契数列元素， 求x=10000000时具体数列.


def fibonacci(x):
    x1=x2=1
    x3=[1,1]    
    while x3[-1]<=x:    
        x1=x2
        x2=x3[-1]
        x3.append(x1+x2)  
    return x3[:-1] 

x =10000000
fibonacci(x)


# 3. 自定义一个正方形的类：
##1. 给出其位置和边长属性
class square:
    """正方形"""
    def __init__(self,t,x):
        self.pos=t 
        self.a=x 

d=square((0,0),1)
d.pos
d.a

##2. 更改其 __init__ 方法，位置和边长属性默认为(0,0)和1
class square:
    """正方形"""
    def __init__(self,t=(0,0),x=1):
        self.pos=t
        self.a=x
d = square()
d.pos
d.a

##3. 更改其 __str__ 方法

class square:
    """正方形"""
    def __init__(self,t=(0,0),x=1):
        self.pos=t
        self.a=x
    def __str__(self):
        print('正方形：', '位置为：', self.pos, '边长为：',self.a)

d = square()
d.__str__()

##4. 定义一个求其面积的方法
class square:
    """正方形"""
    def __init__(self,t=(0,0),x=1):
        self.pos=t
        self.a=x
    def __str__(self):
        print('正方形：', '位置为：', self.pos, '边长为：',self.a)
    def area(self):   # 定义面积
        return self.a**2

d.area()

##5. 定义一个两个正方形距离的方法
class square:
    """正方形"""
    def __init__(self,t=(0,0),x=1):
        self.pos=t
        self.a=x
    def __str__(self):
      print('正方形：', '位置为：', self.pos, '边长为：',self.a)
    def area(self):   # 定义面积
        return self.a**2    
    def dis(self,other):
        return((self.pos[0]-other.pos[0])**2+(self.pos[1]-other.pos[1])**2)**0.5

b=square((2,4),3)
dis(d,b)