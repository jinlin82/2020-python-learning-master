###### 作业内容 #######

###  1. 使用循环数据集

     ### 1. 用 for 循环读入所有csv文件
     ### 2. 把所有读入的csv数据分别转换成一列向量的形式
     ### 3. 创建时间和地区的面板数据的表头
     ### 4. 把表头和41个列向量合并成一个数据框
     ### 5. 对数据框的变量名进行修改为time, dis和 41 个 csv文件的名字
     ### 6. 把最后得到的数据框写出为csv文件

import numpy as np
import pandas as pd 
import os    

csvfiles=os.listdir('../data')   #读取data文件夹里的所有文件名
dat=pd.DataFrame()    #首先建立一个空表
for i in csvfiles:
    temp=pd.read_csv('../data/'+i,encoding='gb2312')       
    temp1=temp.iloc[:,1:].values.T.reshape(180)  #也可以写reshape(-1)，更具一般性；DataFrame转化为array才能reshape
    dat[i[:-4]]=temp1    #删掉文件名后面的.csv作为变量名
dat

## 法一 ##
y=np.arange(2002,2012)
year=pd.Series(np.repeat(y,18));year   #向量的每个元素重复十次
d=temp.district
dis=pd.Series(np.tile(d,10));dis   #整个向量重复十次

dat1=pd.concat([year,dis,dat],axis=1);dat1   #合并前需将array转化为Series;按列合并
dat1.columns.values[:2]=['time','dis']
dat1

dat1.to_csv('./dat1.csv',encoding='gb2312')

## 法二 ##
dis,year=np.meshgrid(d,y)  #meshgrid()函数的功能
dis.shape
year.shape
District=pd.Series(dis.reshape(-1))
Year=pd.Series(year.reshape(-1))
dat2=pd.concat([Year,District,dat],axis=1)
dat2.columns.values[:2]=['time','dis']
dat2
dat2.to_csv('./dat2.csv',encoding='gb2312')


###  2. 自定义函数

     ### 1. 给出一个 list， 求其平均差 $MAD=\frac{1}{n}\sum_{i}^{n}|x_{i}-\bar x_{i}|$

def MAD(x):
    n=len(x)
    x_bar=np.mean(x)
    s=0
    for i in range(n):
        s=s+np.abs(x[i]-x_bar)
    return print(s/n)

MAD([1,1,1,1])
MAD([1,-1,-1,1])
 
     ### 2. 编写一个函数opposite，把向量倒置，对某一向量使用该函数

def opposite(x):
    X=x[::-1]      #向量倒置
    return X

opposite([1,2,3,4])

     ### 3. 编写一个函数shift，把向量元素右移 k 个位置，对某一向量使用该函数

def shift(x,k):
    head=x[-k:]
    tail=x[:-k]
    return head+tail    #list可以直接合并

shift([1,2,3,4,5],2)

     ### 4. 生成一个20行10列的矩阵，把矩阵的每一列倒置，把矩阵的每一行元素向右3个位置

x=np.arange(200).reshape(20,10)
x1=opposite(x)
X=list()
for i in range(20):
    x_i=shift(x1[i,:].tolist(),3)    #tolist()函数将array转化为list
    X.append(x_i)
print(np.array(X))   #list重新转化为array


     ### 5. 编写一个函数 fibonacci ，给定一个正整数x, 生成小于x的所有斐波那契数列元素， 求x=10000000时具体数列.

#运用while循环语句
def fibonacci(x):
    x1=x2=1
    x3=[1,1]    
    while x3[-1]<=x:    #x3数列最后一个元素小于等于
        x1=x2
        x2=x3[-1]
        x3.append(x1+x2)  #数列扩充
    return x3[:-1]   #最后一个超过，故删除

fibonacci(10000000)


###  3. 自定义一个正方形的类：

     ### 1. 给出其位置和边长属性
     
class square:
    '''一个正方形'''
    def __init__(self,t,x):
        self.pos=t
        self.r=x

a=square((0,0),1)
a.pos
a.r

     ### 2. 更改其 __init__ 方法，位置和边长属性默认为(0,0)和1

class square:
    '''一个正方形'''
    def __init__(self,t=(0,0),x=1):   #默认属性
        self.pos=t
        self.r=x

b=square()
b.pos
b.r

     ### 3. 更改其 __str__ 方法

class square:
    '''一个正方形'''
    def __init__(self,t=(0,0),x=1):   #默认设置
        self.pos=t
        self.r=x
    def __str__(self):
        print('正方形:','位置为:',self.pos,'边长为:',self.r)

c=square()
c.__str__

     ### 4. 定义一个求其面积的方法

class square:
    '''一个正方形'''
    def __init__(self,t=(0,0),x=1):   #默认设置
        self.pos=t
        self.r=x
    def __str__(self):
        print('正方形:','位置为:',self.pos,'边长为:',self.r)
    def area(self):
        return self.r**2

d=square((1,1),3)
d.area()

     ### 5. 定义一个两个正方形距离的方法

class square:
    '''一个正方形'''
    def __init__(self,t=(0,0),x=1):   #默认设置
        self.pos=t
        self.r=x
    def __str__(self):
        print('正方形:','位置为:',self.pos,'边长为:',self.r)
    def area(self):
        return self.r**2
    def dis2(self,other):      #求两个正方形的距离
        return ((self.pos[1]-other.pos[1])**2+(self.pos[0]-other.pos[0])**2)**0.5

e=square((1,1),2)
f=square((4,5),3)
e.dis2(f)