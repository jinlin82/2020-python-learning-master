class

class myclass:
    """第一个类"""
    i=123
    def f(xie):
        print("very good")

a=myclass()
type(a)
dir(a)
a.i
a.f()

class circle:
    """一个二维平面的圆"""
    pos=(0,0)
    r=1
b=circle()
type(b)
dir(b)
b.pos
b.r

class circle:
    """一个二维平面的圆"""
    def __init__(self,t,x):
        self.pos=t
        self.r=x
c=circle((2,2),5)
type(c)
dir(c)
c.pos
c.r

class circle:
    """一个二维平面的圆"""
    def __init__(self,t=(0,1),x=3):
        self.pos=t
        self.r=x
    def __str__(self):
        print("圆:","位置为:",self.pos,"半径为:",self.r)

d=circle()
d.pos
d.r

e=circle((3,3),6)
e.pos
e.r
dir(e)
e.__doc__
e.__class__()
e.__str__()

import numpy as np
def c_area(x):
    return np.pi*x.r**2

c_area(e)

class circle:
    """一个二维平面的圆"""
    def __init__(self,t=(0,1),x=3):
        self.pos=t
        self.r=x
    def area(self):
        return np.pi*self.r**2
    def dis2(self,other):
        return ((self.pos[1]-other.pos[1])**2+(self.pos[0]-other.pos[0])**2)**0.5

g=circle((2,2),4)       
dir(g)
g.r
g.area()
type(g.area)
type(c_area)
type(g.pos)

h=circle()
h.pos
h.r
h.area()

m=circle((1,2),2)
n=circle((4,6),3)
m.dis2(n)
n.dis2(m)


BSdata.index
BSdata.columns
BSdata.values
BSdata.keys

BSdata.shape
type(BSdata.shape)
BSdata.shape[0]
BSdata.shape[1]

import numpy as np
A=np.random.randint(0,10,[10,5])
A.shape
A.reshape(25,2)
# shape属性（固有） reshape 方法（对其进行操作）

BSdata["体重"]
BSdata["体重指数"]=BSdata.体重/(BSdata.身高/100)**2
BSdata.columns

round(BSdata[:10],3)
# 后面数字表示保留的小数位数

BSdata.T
# 转置为属性 不是对对象进行操作 不需要（）
BSdata
dir(BSdata)
BSdata.reshape()

B=np.random.randint(0,10,[10,5])
B

A=pd.DataFrame(A)
B=pd.DataFrame(B)
pd.concat([A,B],axis=0)
# 按行合并
pd.concat([A,B],axis=1)
# 按列合并


BSdata.info()
BSdata.支出
BSdata[["支出","身高","体重"]]
# 两个中括号 第一个中括号表示提取BSdata子集，嵌套的中括号表示将变量合并在一起（list） 
BSdata[["学号","课程"]]

BSdata.iloc[1,3]
BSdata.iloc[:3]
BSdata.iloc[0:10,2:4]
BSdata.iloc[0:5,1:4]
BSdata.iloc[50:,4:]
BSdata.iloc[50:,]
BSdata.iloc[:,3]
BSdata.iloc[:,3:4]
# iloc[a,b)
BSdata.iloc[[1,3,7],[1,3,5]]
BSdata.iloc[:10],["性别","支出"]
BSdata.iloc[[:10],["性别","支出"]]
BSdata.loc[:10,["性别","支出"]]
BSdata.loc[3]
BSdata.loc[3:6]
BSdata.iloc[3:6]
BSdata.iloc[3,6]
iloc前后都是数字，loc只能是名字


type(BSdata)

import numpy as np
dat=np.random.randint(0,100,(50,8))
type(dat)
dat[12]
dat[12:15]
dat[12,5]
dat[1:7,3:7]
dat[[1,3,5],4:6]
dat[:,:5]

x=[1,4,7,2,18,45,23,69,84]
x[2:5]
x[[2,5,8]]
# list不支持间隔取数（取子集），pandas，array和Dataframe支持 可将list变成array形
# 式
import numpy as np
np.array(x)[[1,4,6]]
b=np.array(x)
type(b)
b[[1,4,6]]

import pandas as pd
c=pd.array(x)
type(c)
c
c[[1,4,6]]
d=pd.DataFrame(x)
d
type(d)
d.columns
d.index
d.iloc[[1,4,7],:]
d.iloc[2,0]
# DataFrame形式取子集要加iloc[[],:](二维数组)  array不加(一维向量)

# 判断语句
BSdata
BSdata[BSdata["支出"]>30]
BSdata[(BSdata["支出"]>10)&(BSdata['性别']=='女')]
BSdata.支出&BSdata.性别
BSdata.iloc[(BSdata.支出>20)&(BSdata.性别=='女')],['性别','支出']
BSdata.loc[(BSdata.支出>20)&(BSdata.性别=='女'),['性别','支出']]

# 三种形式
BSdata[1:3]
BSdata[] 条件语句
BSdata.iloc[1:3,2:4]
BSdata.loc[] 也可用于条件语句


import numpy as np 
import pandas as pd
x=[1,4,2,7,9,5];x
def xbar(x):
    xm=sum(x)/len(x)
    return xm
xbar(x)    
np.mean(x)

x=np.array([1,4,6,8,3]);x
def ss(x):
    n=len(x)
    ss=sum(x**2)-sum(x)**2/n 
    return ss
ss(x)

def ss2(x):
    n=len(x)
    xm=sum(x)/n 
    ss=sum(x**2)-sum(x)**2/n 
    return[ss,n,xm,x**2]
ss2(x)

ss2(x)[0]
ss2(x)[1]
ss2(x)[2]

type(ss2(x))
type(ss2(x)[2])

BSdata=pd.read_csv('./data/BSdata.csv')
BSdata.性别.value_counts()
BSdata.身高.mean()
BSdata.身高.median()
BSdata.身高.max()-BSdata.身高.min()
BSdata.身高.std()
BSdata.身高.quantile(0.75)-BSdata.身高.quantile(0.25)
##np.quantile(BSdata.身高,0.75)-np.quantile(BSdata.身高,0.25)
BSdata.身高.skew()
BSdata.身高.kurt()

def stats(x):
    stat=[x.count(),x.min(),x.quantile(0.25),x.mean(),x.median(),x.quantile(.75),x.max(),x.max()-x.min(),x.var(),x.std(),x.skew(),x.kurt()]
    stat=pd.Series(stat,index=['Count','Min','Q1(25%)','Mean','Median','Q3(75%)','Max','Range','Var','Std','Skew','Kurt'])
    x.plot(kind='kde')
    return(stat)
stats(BSdata.身高)
stats(BSdata.支出)

##常用绘图函数
import matplotlib.pyplot as plt 
plt.rcParams['font.sans-serif']=['KaiTi']##SimHei黑体
plt.rcParams['axes.unicode_minus']=False##正常显示图中负号
plt.figure(figsize=(5,4))##图形大小

x=['a','b','c','d']
y=[1,4,7,3]
plt.plot(x,y)
plt.bar(x,y)##条形图
plt.pie(y,labels=x) 
plt.hist(BSdata.身高)##频数直方图
plt.hist(BSdata.身高,density=True)##频率直方图
plt.scatter(BSdata.身高,BSdata.体重)##散点图

plt.plot(x,y,linestyle='--',marker='o')
plt.axvline()##垂线
plt.axhline()##水平线
plt.plot(x,y,'o--');plt.axvline(x=1);plt.axhline(y=4)##添加水平线
plt.plot(x,y);plt.text(2,7,'peak point')##在（2，7）处添加峰值
plt.plot(x,y,label='折线');plt.legend()##添加图形名
##误差条图
s=[0.1,0.5,0.2,0.3]
plt.bar(x,y,yerr=s,error_kw={'capsize':5})

#多图
subplot(numrows,numcols,plotnum)
##一行绘制多个图形
plt.subplot(121);plt.bar(x,y);
plt.subplot(122);plt.plot(y);
##一列绘制两个图形
plt.subplot(211);plt.bar(x,y);
plt.subplot(212);plt.plot(y);
##一页绘制两个图形
fig,ax=plt.subplots(1,2,figsize=(15,6))
ax[0].bar(x,y)
ax[1].plot(x,y)
##一页绘制四个图形
fig,ax=plt.subplots(2,2,figsize=(15,12))
ax[0,0].bar(x,y)
ax[0,1].pie(y,labels=x)
ax[1,0].plot(y)
ax[1,1].plot(y,'o--',linewidth=3)

##基于pandas的绘图
##计量数据
BSdata['体重'].plot(kind='line')
BSdata['体重'].plot(kind='hist')
BSdata['体重'].plot(kind='box')##箱线图
BSdata['体重'].plot(kind='density',title='Density')##概率密度曲线图
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(1,3),kind='hist')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(1,3),kind='density')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(3,1),kind='density')

##计数数据
t1=BSdata['开设'].value_counts();t1
pd.DataFrame({'频数':t1,'频率':t1/t1.sum()*100})
t1.plot(kind='bar')
t1.plot(kind='pie')

##数据的分类分析
BSdata['开设'].value_counts()
BSdata.pivot_table(values='学号',index='开设',aggfunc=len)

def tab(x,plot=False):##计数频数表
    f=x.value_counts();f
    s=sum(f);
    p=round(f/s*100,3);p
    t1=pd.concat([f,p],axis=1);
    t1.columns=['例数','构成比'];
    t2=pd.DataFrame({'例数':s,'构成比':100.00},index=['合计'])
    tab=t1.append(t2)
    if plot:
        fig,ax=plt.subplots(1,2,figsize=(15,6))
        ax[0].bar(f.index,f);##条形图
        ax[1].pie(p,labels=p.index,autopct='%1.2f%%')##饼图
    return(round(tab,3))
tab(BSdata.开设,True)

pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts()
pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts().plot(kind='bar')

##自定义计量频率分析函数
def freq(x,bins=10):
    h=plt.hist(x,bins)
    a=h[1][:-1];a
    b=h[1][1:];b
    f=h[0];f
    p=f/sum(f)*100;p
    cp=np.cumsum(p);cp
    freq=pd.DataFrame([a,b,f,p,cp])
    freq.index=['下限','上限','频数','频率(%)','累计频数(%)']
    return(round(freq.T,2))
freq(BSdata.体重)

##二维聚类分析
##二维列联表
pd.crosstab(BSdata.开设,BSdata.课程)
pd.crosstab(BSdata.开设,BSdata.课程,margins=True)
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='index')##各数据占行的比例
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='columns')##列比
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='all').round(3)##保留三位有效数字

##复式条图
t2=pd.crosstab(BSdata.开设,BSdata.课程);t2
t2.plot(kind='bar')##默认为False 表示分段式条形图
t2.plot(kind='bar',stacked=True)##并列式条形图

BSdata.groupby(['性别'])
type(BSdata.groupby(['性别']))
BSdata.groupby(['性别'])['身高'].mean()
BSdata.groupby(['性别'])['身高'].size()
BSdata.groupby(['性别','开设'])['身高'].mean().round(3)

BSdata.groupby(['性别','开设'])['身高'].agg([np.mean,np.std])
BSdata.groupby(['性别'])['身高','体重'].apply(np.mean)
BSdata.groupby(['性别','开设'])['身高','体重'].apply(np.mean)
##区分agg & apply

##多维透视分析
##value_counts()一维表 croostab()二维表 pivot_table()任意维统计表
##计数数据透视分析
BSdata.pivot_table(index=['性别'],values=['学号'],aggfunc=len)
BSdata.pivot_table(index=['性别'],values=['学号','开设'],aggfunc=len)
BSdata.pivot_table(index=['开设'],values=['学号'],columns=['性别'],aggfunc=len)

##计量数据透视分析
BSdata.pivot_table(index=['性别'],values=['身高'],aggfunc=np.mean)
BSdata.pivot_table(index=['性别'],values=['身高'],aggfunc=[np.mean,np.std])
BSdata.pivot_table(index=['性别'],values=['身高','体重'],aggfunc=[np.mean,np.std])
BSdata.pivot_table(index=['性别'],values=['身高','体重'])##默认计算均值

##复合数据透视表
BSdata.pivot_table('学号',['性别','开设'],'课程',aggfunc=len,margins=True,margins_name='合计')
BSdata.pivot_table(['身高','体重'],['性别','开设'],aggfunc=[len,np.mean,np.std])

