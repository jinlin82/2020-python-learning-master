#03beamer

## 条件语句if/else

a=-100
if a < 100:
    print("数值小于100")
else:
    print("数值大于100")

-a if a<0 else a

a=50
if a>100:
    print("数值大于100")
    print(a)
###两个不一样
a=50
if a>100:
    print("数值大于100")
print(a)

print("数值大于100") if a>100 else print(a)

## 循环语句for

for i in range(1,5):
    print(i)

fruits = ['banana', 'apple', 'mango']
for fruit in fruits:
    print('当前水果：',fruit)

for var in BSdata.columns:
    print(var)

import string

string.ascii_lowercase

for i in string.ascii_lowercase:
    print(i)

import numpy as np

x=np.random.randint(-10,10,15);x
for i in x:
    if i>=0:
        print(i**0.5)
    else:
        break

for i in x:
    if i>=0:
        print(i**0.5)
    else:
        continue

## 自定义函数

x=[1,3,6,4,9,7,5,8,2];x

def xbar1(x):
    n=len(x)
    xm=sum(x)/n
    xm

def xbar2(x):
    n=len(x)
    xm=sum(x)/n
    return(xm)

xbar1(x)
xbar2(x)


y=[1,8,3,19,20,-4,5,2,8,47]

len(y)

sum(y)

def mean(Lst):
    return sum(Lst)/len(Lst)

mean(y)

y2=list()
for i in y:
    y2.append(i-1)

def ssq(Lst,x):
    res=list()
    for i in Lst:
        res.append((i-x)**2)
    return res

def var(Lst):
    Lst_mean=mean(Lst)
    return sum(ssq(Lst,Lst_mean))/(len(Lst)-1)

var(y)

x=np.array([1,3,6,4,9,7,5,8,2]);x
def SS1(x):
    n=len(x)
    ss=sum(x**2)-sum(x)**2/n
    return(ss)
SS1(x)

def SS2(x):
    n=len(x)
    xm=sum(x)/n
    ss=sum(x**2)-sum(x)**2/n
    return[x**2,n,xm,ss]
SS2(x)
SS2(x)[0]
SS2(x)[1]
SS2(x)[2]
SS2(x)[3]

type(SS2(x))
type(SS2(x)[3])


import numpy as np
np.var(y,ddof=1)

type(ssq)
type(y)

# 王斌会——Python数据分析基础

## 第3章 Python编程分析基础

###查看数据对象
who

###生成数据对象
x=10.12
who

###删除数据对象
del x
who

###数值型
n=3
n-10 #整数
n #无格式输出，相当于print(n)
print("n=",n)#有格式输出
x=10.234 #实数
print(x)
print("x=%10.5f"%x)

###逻辑型
a=True;a
b=False;b

10>3
10<3

###字符型
s='I love Python';s
s[7]
s[2:6]
s+s
s*2

###list(列表)
list1=[]
list1
list1=['Python',786,2.23,'R',70.2]
list1
list1[0]
list1[1:3]
list1[2:]
list1*2
list1+list1[2:4]

x=[1,3,6,4,9];x
sex=['女','男','男','女','男']
sex
weight=[67,66,83,68,70]
weight

###dictionary(字典)
{} #空字典
dict1={'name':'john','code':6734,'dept':'sales'};dict1
dict1['code']
dict1.keys()
dict1.values()

dict2={'sex':sex,'weight':weight};dict2


## 数值分析库numpy

###一维数组（向量）
np.array([1,2,3,4,5])

np.array([1,2,3,np.nan,5]) #包含缺失值的数组

np.arange(9) #数组序列
np.arange(1,9,0.5) #等差数列
np.linspace(1,9,5) #等距数列

###二维数组（矩阵）
np.array([[1,2],[3,4],[5,6]])

A=np.arange(9).reshape((3,3));A

###数组的维度
A.shape

###空数组
np.empty((3,3))

###零数组
np.zeros((3,3))

###1数组
np.ones((3,3))

###单位阵
np.eye(3)


##数据分析库pandas

###生成序列
import pandas as pd
pd.Series() #生成空序列

###根据列表构建序列
x=[1,3,6,4,9]
s1=pd.Series(x);s1
s2=pd.Series(weight);s2
s3=pd.Series(sex);s3

###序列合并
pd.concat([s2,s3],axis=0)
pd.concat([s2,s3],axis=1)

###序列切片
s1[2]
s3[1:4]

###生成数据框
pd.DataFrame()

###根据列表创建数据框
pd.DataFrame(x)
pd.DataFrame(x,columns=['x'],index=range(5))
pd.DataFrame(weight,columns=['weight'],index=['A','B','C','D','E'])

###根据字典创建数据框
df1=pd.DataFrame({'s1':s1,'s2':s2,'s3':s3});df1

df2=pd.DataFrame({'sex':sex,'weight':weight},index=x);df2

###增加数据框列
df2['weight2']=df2['weight']**2;df2

###删除数据框列
del df2['weight2'];df2

###缺失值处理
df3=pd.DataFrame({'s2':s2,'s3':s3},index=s1);df3
df3.isnull() #若是缺失值则返回True,否则返回False
df3.isnull().sum() #返回每列包含的缺失值的个数
df3.dropna() #直接删除含有缺失值的行，多变量谨慎使用

###数据框排序
df3.sort_index() #按index排序
df3.sort_values(by='s3') #按列值排序

###从剪贴板上读取
BSdata=pd.read_clipboard()
BSdata[:5]

###读取csv格式数据
BSdata=pd.read_csv("BSdata.csv",encoding='utf-8')
BSdata[6:9]

###读取excel格式数据
BSdata=pd.read_excel('DaPy_data.xlsx','BSdata');BSdata[-5:]

###pandas数据集的保存
BSdata.to_csv('BSdata1.csv')
BSdata.to_excel('BSdata1.xlsx',index=False)

###数据框显示
BSdata.info() #数据框信息
BSdata.head()
BSdata.tail()

###数据框列名（变量名）
BSdata.columns

###数据框行名（样品名）
BSdata.index

###数据框维度
BSdata.shape
BSdata.shape[0] #数据框行数
BSdata.shape[1] #数据框列数

###数据框值（数组）
BSdata.values[:5]

###选取变量
BSdata.身高
BSdata[['身高','体重']]

BSdata.iloc[:,2]
BSdata.iloc[:,2:4]

###提取样品
BSdata.loc[3]
BSdata.loc[3:5]

###选取观测与变量
BSdata.loc[:3,['身高','体重']]
BSdata.iloc[:3,:5]

###条件选取
BSdata[BSdata['身高']>180]
BSdata[(BSdata['身高']>180)&(BSdata['体重']<80)]

###生成新的数据框
BSdata['体重指数']=BSdata['体重']/(BSdata['身高']/100)**2
round(BSdata[:5],2)

###数据框的合并
pd.concat([BSdata.身高,BSdata.体重],axis=0)
pd.concat([BSdata.身高,BSdata.体重],axis=1)
BSdata.iloc[:3,:5].T

##class
class Myclass:
    """第一个类"""
    i=1234 #属性
    def f(self):
        print('hello world')
a=Myclass()
type(a)
dir(a)
a.i
a.f()

class circle1:
    """一个二维平面中的圆"""
    pos=(0,0)
    r=1
b=circle1()
type(b)
dir(b)
b.pos
b.r

class circle:
    """一个二维平面中的圆"""
    def __init__(self,t,x):
        self.pos=t
        self.r=x

c=circle((1,1),10)
c.pos
c.r

class circle:
    """一个二维平面中的圆"""
    def __init__(self,t=(0,0),x=1):
        self.pos=t
        self.r=x
d=circle()
d.pos
d.r

g=circle((1,2),20)
dir(g)
g.__doc__
g.__str__()
print(g)

class circle:
    """一个二维平面中的圆"""
    def __init__(self,t=(0,0),x=1):
        self.pos=t
        self.r=x
    def __str__(self):
        print('圆：', '位置为：', self.pos, '半径为：',self.r)

dir(g)
g.__doc__
g.__str__()

def c_area(x):
    return np.pi*x.r**2
c_area(g)

class circle:
    """一个二维平面中的圆"""
    def __init__(self,t=(0,0),x=1):
        self.pos=t
        self.r=x
    def __str__(self):
        print('圆：', '位置为：', self.pos, '半径为：',self.r)
    def area(self):
        return np.pi*self.r**2
    def dis2(self,other):
        return ((self.pos[1]-other.pos[1])**2+(self.pos[0]-other.pos[0])**2)**0.5

#######class里的函数就是方法
h=circle((1,1),4)
dir(h)
h.area()
type(h.area)
type(c_area)
type(h.pos)

k=circle((3,8),3)
h.dis2(k)

## 第4章 数据的探索性分析

### 基本描述统计量
BSdata.describe()
BSdata[['性别','开设','课程','软件']].describe() #数值型和分类型数据要分开，不然只计算数值型

### 计数数据汇总分析
T1=BSdata.性别.value_counts();T1
T1/sum(T1)*100

### 计量数据汇总分析
BSdata.身高.mean()
BSdata.身高.median()
BSdata.身高.max()-BSdata.身高.min()
BSdata.身高.var()
BSdata.身高.std()
BSdata.身高.quantile(0.75)-BSdata.身高.quantile(0.25)
BSdata.身高.skew()
BSdata.身高.kurt()

def stats(x):
    stat=[x.count(),x.min(),x.quantile(0.25),x.mean(),x.median(),x.quantile(0.75),x.max(),x.max()-x.min(),x.var(),x.std(),x.skew(),x.kurt()]
    stat=pd.Series(stat,index=['Count','Min','Q1(25%)','Mean','Median','Q3(75%)','Max','Range','Var','Std','Skew','Kurt'])
    x.plot(kind='kde')   #拟合核密度kde曲线
    return(stat)
stats(BSdata.身高)
stats(BSdata.支出)

### 常用的绘图函数
import matplotlib.pyplot as plt #基本绘图包
plt.rcParams['font.sans-serif']=['KaiTi'] #SimHei黑体
plt.rcParams['axes.unicode_minus']=False #正常显示图中负号
plt.figure(figsize=(5,4))

x=['A','B','C','D','E','F','G']
y=[1,4,7,3,2,5,6]
plt.bar(x,y) #条形图
plt.pie(y,labels=x) #饼图
plt.plot(x,y) #线图
plt.hist(BSdata.身高) #频数直方图
plt.hist(BSdata.身高,density=True) #频率直方图
plt.scatter(BSdata.身高,BSdata.体重) #散点图

plt.plot(x,y,c='red');plt.ylim(0,8)
plt.xlabel('names');plt.ylabel('values')
plt.xticks(range(len(x)),x) #标题、标签、标尺及颜色

plt.plot(x,y,linestyle='--',marker='o') #线型和符号

plt.plot(x,y,'o--');plt.axvline(x=1);plt.axhline(y=4) #绘图函数附加图形

plt.plot(x,y);plt.text(2,7,'peak point') #文字函数

plt.plot(x,y,label=u'折线');plt.legend() #图例

s=[0.1,0.4,0.7,0.3,0.2,0.5,0.6]
plt.bar(x,y,yerr=s,error_kw={'capsize':5}) #误差条图

'''一行绘制两个图形'''
plt.subplot(121);plt.bar(x,y)
plt.subplot(122);plt.plot(y)

'''一列绘制两个图形'''
plt.subplot(211);plt.bar(x,y)
plt.subplot(212);plt.plot(y)

'''一页绘制两个图形'''
fig,ax=plt.subplots(1,2,figsize=(15,6))
ax[0].bar(x,y)
ax[1].plot(x,y)

'''一页绘制四个图形'''
fig,ax=plt.subplots(2,2,figsize=(15,12))
ax[0,0].bar(x,y)
ax[0,1].pie(y,labels=x)
ax[1,0].plot(x,y)
ax[1,1].plot(y,'.-',linewidth=3)

###基于pandas的绘图

####计量数据
BSdata['体重'].plot(kind='line')
BSdata['体重'].plot(kind='hist')
BSdata['体重'].plot(kind='box')
BSdata['体重'].plot(kind='density',title='Density')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(1,3),kind='box')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(1,3),kind='density')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(3,1),kind='density')

####计数数据
T1=BSdata['开设'].value_counts();T1
pd.DataFrame({'频数':T1,'频率':T1/T1.sum()*100})
T1.plot(kind='bar')
T1.plot(kind='pie')

###数据的分类分析

####一维频数分析
BSdata['开设'].value_counts()
BSdata.pivot_table(values='学号',index='开设',aggfunc=len) #计数数据频数分布

def tab(x,plot=False):
    f=x.value_counts();f
    s=sum(f)
    p=round(f/s*100,3);p
    T1=pd.concat([f,p],axis=1)
    T1.columns=['例数','构成比']
    T2=pd.DataFrame({'例数':s,'构成比':100.00},index=['合计'])
    Tab=T1.append(T2)
    if plot:
        fig,ax=plt.subplots(2,1,figsize=(8,15))
        ax[0].bar(f.index,f)
        ax[1].pie(p,labels=p.index,autopct='%1.2f%%')
    return(round(Tab,3))

tab(BSdata.开设,True)

pd.cut(BSdata.身高,bins=10).value_counts() #计量数据频数分布
pd.cut(BSdata.身高,bins=10).value_counts().plot(kind='bar')
pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts()
pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts().plot(kind='bar')

def freq(x,bins=10):
    H=plt.hist(x,bins)
    a=H[1][:-1];a
    b=H[1][1:];b
    f=H[0];f
    p=f/sum(f)*100;p
    cp=np.cumsum(p);cp
    Freq=pd.DataFrame([a,b,f,p,cp])
    Freq.index=['[下限','上限)','频数','频率(%)','累计频数(%)']
    return(round(Freq.T,2))
freq(BSdata.体重)

#### 二维集聚分析

##### 计数数据的列联表
pd.crosstab(BSdata.开设,BSdata.课程) 
pd.crosstab(BSdata.开设,BSdata.课程,margins=True)
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='index') #各数据占行的比例
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='columns') #各数据占列的比例
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='all').round(3) #各数据占总和的构成比例

T2=pd.crosstab(BSdata.开设,BSdata.课程);T2
T2.plot(kind='bar') #复式条图
T2.plot(kind='bar',stacked=True) #并列式条形图，默认为false

#####计量数据的集聚表
BSdata.groupby(['性别']) #按列分组
type(BSdata.groupby(['性别'])) #groupby()生成的是一个中间分组变量，为GroupBy类型

BSdata.groupby(['性别'])['身高'].mean()
BSdata.groupby(['性别'])['身高'].size()
BSdata.groupby(['性别','开设'])['身高'].mean() #按分组统计

BSdata.groupby(['性别'])['身高'].agg([np.mean,np.std]) #应用agg()，仅作用于指定的列

BSdata.groupby(['性别'])['身高','体重'].apply(np.mean)
BSdata.groupby(['性别','开设'])['身高','体重'].apply(np.mean) #应用apply()，应用于dataframe的各个列

#### 多维透视分析

#####计数数据透视分析
BSdata.pivot_table(index=['性别'],values=['学号'],aggfunc=len)
BSdata.pivot_table(values=['学号'],index=['性别','开设'],aggfunc=len)
BSdata.pivot_table(index=['开设'],values=['学号'],columns=['性别'],aggfunc=len)

#####计量数据透视分析
BSdata.pivot_table(index=['性别'],values=['身高'],aggfunc=np.mean)
BSdata.pivot_table(index=['性别'],values=['身高'],aggfunc=[np.mean,np.std])
BSdata.pivot_table(index=['性别'],values=['身高','体重'])#默认计算均值

#####复合数据透视分析
BSdata.pivot_table('学号',['性别','开设'],'课程',aggfunc=len,margins=True,margins_name='合计')
BSdata.pivot_table(['身高','体重'],['性别','开设'],aggfunc=[len,np.mean,np.std])

#pandas
import pandas as pd
import numpy as np

##object creation
pd.Series([1,2,5,np.nan,8])

dates=pd.date_range('20130101',periods=6)
df=pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))

df2=pd.DataFrame(np.random.randn(5,4),index=pd.date_range('20130101',periods=5),columns=list('ABCD'))
df2.loc[2:3] #错误，必须都是使用标签，而不是数字
df2.loc['20130102':'20130104']
df2.loc[:,['A','C']] #使用标签（label）选取

df2.iloc[2]
df2.iloc[2:3]
df2.iloc[[0,2,4]]
df2.iloc[[0,2,4],:3] #使用位置选取

##逻辑值下标
###using a single column's values to select data
df[df.A>0]
df[df>0]
df2.loc[df2['A']>0,'A':'C']
df2.iloc[list(df2['A']>0),0:3]

##indexing with isin
s=pd.Series(np.arange(5),index=np.arange(5)[::-1],dtype='int64')
s[s.isin([2,4,6])]

df=pd.DataFrame({'vals':[1,2,3,4],'ids':['a','b','f','n'],'id2':['a','n','c','n']})
values=['a','b',1,3]
df.isin(values)

values={'ids':['a','b'],'vals':[1,3]}
df.isin(values)

values={'ids':['a','b'],'id2':['a','c'],'vals':[1,3]}
row_mask=df.isin(values).all(1)
df[row_mask]

## the where() method
s[s>0]
s.where(s>0)
df.where(df<0,-df)   ?

## duplicate data
df2=pd.DataFrame({'a':['one','one','two','two','two','three','four'],'b':['x','y','x','y','x','x','x'],'c':np.random.randn(7)})
df2.duplicated('a')
df2.duplicated('a',keep='last')
df2.duplicated('a',keep=False)
df2.drop_duplicates('a')
df2.drop_duplicates('a',keep='last')
df2.drop_duplicates('a',keep=False)

df2.duplicated(['a','b'])
df2.drop_duplicates(['a','b']) #pass a list of columns to identify duplications

##字符处理
s=pd.Series(['A','B','C','Aaba','Baca',np.nan,'CABA','dog','cat'])
s.str.lower()

##concat
df=pd.DataFrame(np.random.randn(10,4));df
pieces=[df[:3],df[3:7],df[7:]];pieces
pd.concat(pieces)

##append
df=pd.DataFrame(np.random.randn(8,4),columns=['A','B','C','D']);df
s=df.iloc[3];s
df.append(s)
df.append(s,ignore_index=True)

##groupby
df=pd.DataFrame({'A':['foo','bar','foo','bar','foo','bar','foo','foo'],'B':['one','one','two','three','two','two','one','three'],'C':np.random.randn(8),'D':np.random.randn(8)})
df.groupby('A').sum()
df.groupby(['A','B']).sum()
df.groupby(['A','B']).agg([np.sum,np.mean,np.std])
df.groupby(['A','B']).agg({'C':np.sum,'D':lambda x: np.std(x,ddof=1)})

##group plotting
np.random.seed(1234)
df=pd.DataFrame(np.random.randn(50,2))
df['g']=np.random.choice(['A','B'],size=50)
df.loc[df['g']=='B',1]+=3
df.groupby('g').boxplot()

##reshaping
###使用pivot()方法
import pandas.util.testing as tm; tm.N = 3
def unpivot(frame):
    N, K = frame.shape
    data = {'value' : frame.values.ravel('F'),'variable' : np.asarray(frame.columns).repeat(N),'date' : np.tile(np.asarray(frame.index), K)} 
    return pd.DataFrame(data, columns=['date', 'variable', 'value'])
df = unpivot(tm.makeTimeDataFrame())

df[df['variable'] == 'A']

df.pivot(index='date', columns='variable', values='value') 
df['value2'] = df['value']*2
df.pivot('date', 'variable')
pivoted = df.pivot('date', 'variable')
pivoted['value2']

## pivot tables
df = pd.DataFrame({"A": ["foo", "foo", "foo","foo", "foo", "bar", "bar", "bar", "bar"],"B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],"C": ["small", "large", "large", "small", "small", "large", "small", "small", "large"],"D": [1, 2, 2, 3, 3, 4, 5, 6, 7], "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]});df

table = pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'], aggfunc=np.sum);table

table = pd.pivot_table(df, values=['D', 'E'], index=['A', 'C'], aggfunc={'D': np.mean,'E': np.mean})
table = pd.pivot_table(df, values=['D', 'E'], index=['A', 'C'], aggfunc={'D': np.mean,'E': [min, max, np.mean]})

## cross tabulations
foo, bar, dull, shiny, one, two = 'foo','bar', 'dull', 'shiny' , 'one', 'two'
a = np.array([foo, foo, bar, bar, foo, foo],dtype=object) 
b = np.array([one, one, two, one, two, one], dtype=object) 
c = np.array([dull, dull, shiny, dull, dull, shiny], dtype=object)

pd.crosstab(a, [b, c], rownames=['a'],colnames=['b', 'c'])

df = pd.DataFrame({'A': [1, 2, 2, 2, 2], 'B': [3, 3, 4, 4, 4], 'C': [1, 1, np.nan, 1, 1]});df

pd.crosstab(df.A, df.B)
pd.crosstab(df['A'], df['B'], normalize=True)
pd.crosstab(df['A'], df['B'],normalize='columns')
pd.crosstab(df.A, df.B, values=df.C, aggfunc=np.sum, normalize=True, margins=True)

##cut function
ages=np.array([10,15,13,12,23,25,28,59,60])
pd.cut(ages,bins=3)
pd.cut(ages,bins=[0,18,35,70])

##dummy variables: get_dummies()
df = pd.DataFrame({'key': list('bbacab'), 'data1': range(6)});df
pd.get_dummies(df['key'])
dummies = pd.get_dummies(df['key'], prefix='key')

df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['c', 'c', 'b'], 'C': [1, 2, 3]})
pd.get_dummies(df, columns=['A'])

##创建分类
s = pd.Series(["a","b","c","a"],dtype="category")
df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})

df["grade"] = df["raw_grade"].astype("category")
df["grade"].cat.categories = ["very good", "good", "very bad"]
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
df.sort_values(by="grade")
df.groupby("grade").size()

##plot()
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000',periods=1000));ts
ts = ts.cumsum()
ts.plot()

df3 = pd.DataFrame(np.random.randn(1000, 2), columns=['B', 'C' ]).cumsum();df3
df3['A'] = pd.Series(list(range(len(df3)))) 
df3.plot(x='A', y='B')
