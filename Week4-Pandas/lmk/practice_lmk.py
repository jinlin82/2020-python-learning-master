# 控制语句
## 循环语句 for
for i in range(1,5):
    print(i)

fruits=['banana','apple','mango']
for fruit in fruits:
    print('当前水果：',fruit)

import pandas as pd
BSdata=pd.read_csv("../lmk/BSdata.csv",encoding='utf-8')
for var in BSdata.columns:
    print(var)

## 条件语句 if/else
a=-100
if a<100:
    print("数值小于100")
else:
    print("数值大于100")

-a if a<0 else a

# 自定义函数
## 定义函数的语法
def 函数名（参数1，参数2，...）:
    函数体
    return 语句

import numpy as np
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

np.mean(x)

# 面向对象
X=np.array([1,3,6,4,9,7,5,8,2]);X
def SS1(x):
    n=len(x)
    ss=sum(x**2)-sum(x)**2/n
    return(ss)
SS1(X)

def SS2(x):
    n=len(x)
    xm=sum(xm)/n
    ss=sum(x**2)-sum(x)**2/n
    return(x**2,n,xm,ss)
SS2(X)

SS2(X)[0]
SS2(X)[1]
SS2(X)[2]
SS2(X)[3]

type(SS2(X))
type(SS2(X)[3])

# class
class circle:
    """一个二维平面中的圆"""
    def __init__(self,t=(0,0),x=1):
        self.pos=t
        self.r=x
    def __str__(self):
        return print('圆：','位置为：',self.pos,'半径为：',self.r)
    def area(self):
        return np.pi*self.r**2
    def dis2(self,other):
        return((self.pos[1]-other.pos[1])**2+(self.pos[0]-other.pos[0])**2)**0.5

h=circle((1,1),4)
dir(h)
h.area()
type(h.area)
type(h.pos)
type(h.r)

k=circle((3,8),3)
h.dis2(k)

# 描述统计
BSdata.describe()
BSdata[['性别','开设','课程','软件']].describe()

T1=BSdata.性别.value_counts();T1 # 频数
T1/sum(T1)*100 # 频率

BSdata.身高.mean()
BSdata.身高.median()
BSdata.身高.max()-BSdata.身高.min()
BSdata.身高.var()
BSdata.身高.std()
BSdata.身高.quantile(0.75)-BSdata.身高.quantile(0.25) # 四分间距
BSdata.身高.skew() # 偏度
BSdata.身高.kurt() # 峰度

def stats(x):
    stat=[x.count(),x.min(),x.quantile(0.25),x.mean(),x.median(),x.quantile(0.75),x.max(),x.max()-x.min(),x.var(),x.std(),x.skew(),x.kurt()]
    stat=pd.Series(stat,index=['Count','Min','Q1(25%)','Mean','Median','Q3(75%)','Max','Range','Var','Std','Skew','Kurt'])
    x.plot(kind='kde') # 拟合核密度曲线
    return(stat)
stats(BSdata.身高)
stats(BSdata.支出)

# 基本绘图命令
import matplotlib.pyplot as plt # 基本绘图包
plt.rcParams['font.sans-serif']=['KaiTi'] # SimHei黑体
plt.rcParams['axes.unicode_minus']=False # 正常显示图中符号
plt.figure(figsize=(5,4)) # 图形大小

# 条图
X=['A','B','C','D','E','F','G']
Y=[1,4,7,3,2,5,6]
plt.bar(X,Y)

# 饼图
plt.pie(Y,labels=X)

# 线图
plt.plot(X,Y)

# 直方图
plt.hist(BSdata.身高) # 频数直方图
plt.hist(BSdata.身高,density=True) # 频率直方图

# 散点图
plt.scatter(BSdata.身高,BSdata.体重)

# 图形参数设置
plt.plot(X,Y,c='red') # 颜色
plt.ylim(0,8) # 纵轴范围
plt.xlabel('names') # 横轴名称
plt.ylabel('values') # 纵轴名称
plt.xticks(range(len(X)),X) # 横轴刻度

plt.plot(X,Y,linestyle='--',marker='o') # 线型和符号

plt.plot(X,Y,'o--')
plt.axvline(x=1) # 在纵坐标y处画垂线
plt.axhline(y=4) # 在横坐标x处画水平线

plt.plot(X,Y)
plt.text(2,7,'peak point') # 在(x,y)处添加用labels指定的文字

plt.plot(X,Y,label='折线')
plt.legend() # 图例

s=[0.1,0.4,0.7,0.3,0.2,0.5,0.6]
plt.bar(X,Y,yerr=s,error_kw={'capsize':5}) # 误差条图

# 多图 subplot(numRows,numCols,plotNum)
## 一行绘制两个图形
plt.subplot(121);plt.bar(X,Y)
plt.subplot(122);plt.plot(Y)

## 一列绘制两个图形
plt.subplot(211);plt.bar(X,Y)
plt.subplot(212);plt.plot(Y)

## 一页绘制两个图形
fig,ax=plt.subplots(1,2,figsize=(15,6))
ax[0].bar(X,Y)
ax[1].plot(X,Y)

## 一页绘制四个图形
fig,ax=plt.subplots(2,2,figsize=(15,12))
ax[0,0].bar(X,Y)
ax[0,1].pie(Y,labels=X)
ax[1,0].plot(Y)
ax[1,1].plot(Y,'.-',linewidth=3)

# 基于pandas的绘图
## 计量数据
BSdata['体重'].plot(kind='line')
BSdata['体重'].plot(kind='hist')
BSdata['体重'].plot(kind='box')
BSdata['体重'].plot(kind='density',title='Density')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(1,3),kind='box')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(1,3),kind='density')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(3,1),kind='density')

## 计数数据
T1=BSdata['开设'].value_counts;T1
pd.DataFrame({'频数':T1,'频率':T1/T1.sum()*100}) #？T1是method

T1.plot(kind='bar')
T1.sort_values().plot(kind='bar')
T1.plot(kind='pie') #？同上

# 一维频数分析
## 计数频数分析
BSdata['开设'].value_counts()
Bsdata.pivot_table(values='学号',index='开设',aggfunc=len) #？invalid

def tab(x,plot=False): # 计数频数表
    f=x.value_counts();f
    s=sum(f)
    p=round(f/s*100,3);p
    T1=pd.concat([f,p],axis=1)
    T1.columns=['例数','构成比']
    T2=pd.DataFrame({'例数':s,'构成比':100.00},index=['合计'])
    Tab=T1.append(T2)
    if plot:
        fig,ax=plt.subplots(1,2,figsize=(15,6))
        ax[0].bar(f.index,f) # 条图
        ax[1].pie(p,labels=p.index,autopct='%1.2f%%') #饼图
    return(round(Tab,3))
tab(BSdata.开设,True)

## 计量频数分析
pd.cut(BSdata.身高,bins=10).value_counts() # 先用cut分组再统计
pd.cut(BSdata.身高,bins=10).value_counts().plot(kind='bar')

pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts()
pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts().plot(kind='bar')

import numpy as np
def freq(X,bins=10): # 计量频数表与直方图
    H=plt.hist(X,bins)
    a=H[1][:-1];a
    b=H[1][1:];b
    f=H[0];f
    p=f/sum(f)*100;p
    cp=np.cumsum(p);cp
    Freq=pd.DataFrame([a,b,f,p,cp])
    Freq.index=['[下限','上限)','频数','频率(%)','累计频数(%)']
    return(round(Freq.T,2))
freq(BSdata.体重)

# 二维集聚分析
## 计数数据
### 列联表
pd.crosstab(BSdata.开设,BSdata.课程)
pd.crosstab(BSdata.开设,BSdata.课程,margins=True) # 行列合计
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='index') # 各数据占行的比例
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='columns') # 各数据占列的比例
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='all').round(3) # 各数据占总和的比例

### 复式条图
T2=pd.crosstab(BSdata.开设,BSdata.课程);T2
T2.plot(kind='bar') # 分段式条图
T2.plot(kind='bar',stacked=True) # 并列式条图
# ？？？中文图例无法显示

## 计量数据
BSdata.groupby(['性别']) # 按列分组
type(BSdata.groupby(['性别']))

BSdata.groupby(['性别'])['身高'].mean() # 按分组统计
BSdata.groupby(['性别'])['身高'].size()
BSdata.groupby(['性别','开设'])['身高'].mean()

BSdata.groupby(['性别'])['身高'].agg([np.mean,np.std]) # agg()作用于各个列

BSdata.groupby(['性别'])['身高'].apply(np.mean)
BSdata.groupby(['性别','开设'])['身高','体重'].apply(np.mean) # apply()作用于指定列

# 多维透视分析
## 计数数据
BSdata.pivot_table(index='性别',values='学号',aggfunc=len)
BSdata.pivot_table(index=['性别','开设'],values='学号',aggfunc=len)
BSdata.pivot_table(index='开设',values='学号',columns='性别',aggfunc=len)

## 计量数据
BSdata.pivot_table(index='性别',values='身高',aggfunc=np.mean)
BSdata.pivot_table(index='性别',values='身高',aggfunc=[np.mean,np.std])
BSdata.pivot_table(index='性别',values=['身高','体重']) # 默认计算均值

## 复合数据
BSdata.pivot_table('学号',['性别','开设'],'课程',aggfunc=len,margins=True,margins_name='合计')
BSdata.pivot_table(['身高','体重'],['性别','开设'],aggfunc=[len,np.mean,np.std])