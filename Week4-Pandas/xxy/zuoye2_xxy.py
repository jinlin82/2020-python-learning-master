##* 作业内容
##1. 使用循环数据集，
##. 用 for 循环读入所有csv文件
import pandas as pd
import numpy as np
gdp=pd.read_csv('./data1/gdp.csv',encoding="utf-8")
gdp.columns
gdp.T.columns
gdp.T.index
gdp.values
gdp.keys
gdp=pd.read_csv('./data1/gdp.csv',encoding='gb2312')
gdp.iloc[:,1:].values.T.reshape(-1)
##数据转置 180个数据


##2. 把所有读入的csv数据分别转换成一列向量的形式
##python directory files name
import pandas as pd
import numpy as np
import os
csvfiles=os.listdir('./data1')
dat=pd.DataFrame()
for i in csvfiles:
   temp=pd.read_csv('./data1/'+i,encoding='utf_8')
   temp1=temp.iloc[:,1:].values.T.reshape(-1)
   dat[i[:-4]]=temp1
   ##reshape(-1) -1表示自动识别总数据
dat

###3. 创建时间和地区的面板数据的表头
temp=pd.read_csv('./data1/'+i,encoding='utf_8')
temp1=temp.iloc[:,1:].values.T.reshape(-1)
temp1=temp.iloc[:,1:]
temp1.columns
np.repeat(temp1.columns,18)
temp.district

##
year=np.arange(2002,2012)
np.repeat(year,18)
dis=temp.district
np.repeat(dis,10)
np.tile(dis,10)
## 或者 np.repeat(dis.values.reshape((18,1)),10,axis=1).T.reshape(-1)
type(dis)

year=np.arange(2002,2012);year
dis=temp.district;dis
dat['year']=np.repeat(year,18)
dat['dis']=np.tile(dis,10)##list(dis.values)*10或者np.array(dis) 将series降为数组
dat
type(dat)


year=np.arange(2002,2012)
dis=temp.district
Year=pd.DataFrame(np.repeat(year,18),columns=['Year'])
District=pd.DataFrame(np.tile(dis,10),columns=['Dis'])
dat2=pd.concat((Year,District,dat),axis=1);dat2

Year=pd.Series(np.repeat(year,18))
District=pd.Series(np.tile(dis,10))
dat2=pd.concat((Year,District,dat),axis=1)
dat2.columns.values[0:2]=['Year','Dis']
##change pandas cloumn name
##带有values
dat2
dat2.to_csv('./dat.csv',encoding='utf-8')##或者'utf—8'

###另一种方式
District,Year=np.meshgrid(dis,year)
District.reshape(-1)
Year.reshape(-1)
Year=pd.Series(Year.reshape(-1))
District=pd.Series(District.reshape(-1))
dat3=pd.concat((Year,District,dat),axis=1)
dat3.columns.values[0:2]=['Year','Dis']

##h=pd.DataFrame({'Year':Year,'Dis':District})
##pd.concat((h,dat),axis=1)


##4. 把表头和41个列向量合并成一个数据框
Year=pd.Series(np.repeat(year,18))
District=pd.Series(np.tile(dis,10))
dat2=pd.concat((Year,District,dat),axis=1)
dat2

##5. 对数据框的变量名进行修改为time, dis和 41 个 csv文件的名字
dat2.columns.values[0:2]=['Year','Dis']
dat2
##6. 把最后得到的数据框写出为csv文件
dat2.to_csv('./dat.csv',encoding='utf-8')


##2. 自定义函数
##1. 给出一个 list， 求其平均差 $MAD=\frac{1}{n}\sum_{i}^{n}|x_{i}-\bar x_{i}|$
def MAD(x):
    n=len(x)
    a=list()
    for i in x:
        x_mean=np.mean(x)
        a.append(np.abs(i-x_mean))
        m=sum(a)
    return m/n
b=[1,2,3];b
MAD(b)

##
def mad(x):
    n=len(x)
    x_mean=np.mean(x)
    s=0
    for i in range(n):
        s=s+np.abs(x[i]-x_mean)
    return s/n 
b=[1,2,3];b
mad(b)


##2. 编写一个函数opposite，把向量倒置，对某一向量使用该函数
def opposite(x):
   return x[::-1]
m=np.random.randint(0,10,10);m
opposite(m)

##
m=[1,2,3]
m.reverse()
m


##3. 编写一个函数shift，把向量元素右移 k 个位置，对某一向量使用该函数
def shift(x,k):
    return list(x[-k:])+list(x[:-k])
    # return list(x[len(x)-k:])+list(:x[len(x)-k])
x=[2,4,7,1,9,5]
shift(x,2)


###4. 生成一个20行10列的矩阵，把矩阵的每一列倒置，把矩阵的每一行元素向右3个位置
a=np.random.randint(0,10,(20,10));a
def new(x):
  b=opposite(a)
  c=list()
  for i in b:
     i=shift(i,3)
     c.append(i)
  return np.array(c)
new(a)


##5. 编写一个函数 fibonacci ，给定一个正整数x, 生成小于x的所有斐波那契数列元素， 求x=10000000时具体数列.
def fibonacci(x):
  a,b=0,1
  while a<x:
    print(a)
    a,b = b,a+b
fibonacci(10000000)

##
x1=0
x2=1
x3=x1+x2
while x3<1000:
   x1=x2
   x2=x3
   x3=x1+x2
   print(x2)

##
def fibo(x):
   x1=0
   x2=1
   x3=[0,1]
   x3.append(x1+x2)
   while x3[-1]<=x:
      x1=x2
      x2=x3[-1]
      x3.append(x1+x2)
   return x3[:-1]
fibo(1000)


##3. 自定义一个正方形的类：
##1. 给出其位置和边长属性
class square:
   """一个正方形"""
   pos=(1,1)
   r=2
a=square() 
dir(a)
a.pos
a.r

##2. 更改其 __init__ 方法，位置和边长属性默认为(0,0)和1
class square:
   """一个正方形"""
   def __init__(self,m=(0,0),n=1):
      self.pos=m
      self.r=n
b=square()
type(b)
b.pos
b.r

##3. 更改其 __str__ 方法
class square:
   """一个正方形"""
   def __init__(self,m=(0,0),n=1):
      self.pos=m
      self.r=n
   def __str__(self):
      print("正方形:","位置为:",self.pos,"半径为:",self.r)

c=square((2,2),5)
c.pos
c.r
c.__doc__
c.__str__()

##4. 定义一个求其面积的方法
def area(x):
   return x.r**2

d=square((3,4),6)
area(d)


##5. 定义一个两个正方形距离的方法
class square:
   """一个正方形"""
   def __init__(self,m=(0,0),n=1):
      self.pos=m
      self.r=n
   def dis(self,other):
      return ((self.pos[1]-other.pos[1])**2+(self.pos[0]-other.pos[0])**2)**0.5
x=square((1,2),3)
y=square((4,6),7)
x.dis(y)

dis(x,y)

