import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
BSdata = pd.read_csv('./BSdata.csv')
BSdata.describe()

BSdata[['性别','开设','课程','软件']].describe()

T1 = BSdata.性别.value_counts();T1
T1/sum(T1)*100
#BSdata.crosstab()
BSdata.性别.value_counts(normalize=True)*100

BSdata.身高.mean()
BSdata.身高.median()
BSdata.身高.max()-BSdata.身高.min()
BSdata.身高.var()
BSdata.身高.std()
BSdata.身高.quantile(0.75)-BSdata.身高.quantile(0.25)
BSdata.身高.skew()
BSdata.身高.kurt()

def stats(x):
    stats = [x.count(),x.min(),x.quantile(0.25),x.mean(),x.median(),x.quantile(0.75),x.max(),x.max()-x.min(),x.var(),x.std(),x.skew(),x.kurt()]
    stats = pd.Series(stats,index=['Count','min','Q1(25%)','Mean','Median','Q3(75%)','Max','Range','Var','Std','Skew','Kurt'])
    return(stats)
stats(BSdata.身高)
stats(BSdata.支出)

import matplotlib.pyplot as plt 
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(6,5))
plt.show()

X = ['A','B','C','D','E','F','G']
Y = [1,4,7,3,2,5,6]
plt.bar(X,Y) #条图
plt.show()
plt.savefig('abc',format='pdf')

plt.pie(Y,labels=X)#饼图
plt.show()

plt.plot(X,Y)
plt.show()

#直方图hist
plt.hist(BSdata.身高)
plt.show()

plt.hist(BSdata.身高,density=True)
plt.show()

#scatter散点图
plt.scatter(BSdata.身高,BSdata.体重)
plt.show()

plt.ylim(0,8)
plt.xlabel('names');plt.ylabel('values')
plt.xticks(range(len(X)),X)
plt.show()

plt.plot(X,Y,linestyle='--',marker='o')
plt.show()

plt.plot(X,Y,'o--')
plt.axvline(x=1)
plt.axhline(y=4)
plt.show()

plt.plot(X,Y)
plt.text(2,7,'peakpoint')
plt.show()

plt.plot(X,Y,label=u'折线')

s = [0.1,0.4,0.7,0.3,0.2,0.5,0.6]
plt.bar(X,Y,yerr=s,error_kw={'capsize':5})
plt.show()

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.bar(X,Y)
plt.subplot(1,2,2)
plt.plot(Y)
plt.show()

fig,ax = plt.subplots(1,2,figsize=(14,6))
ax[0].bar(X,Y)
ax[1].plot(X,Y)
plt.show()

fig,ax = plt.subplots(2,2,figsize=(5,10))
ax[0,0].bar(X,Y)
ax[0,1].pie(Y,labels=X)
ax[1,0].plot(Y)
ax[1,1].plot(Y,'-',linewidth=3)
plt.show()

plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
BSdata['体重'].plot(kind='line')
plt.subplot(2,2,2)
BSdata['体重'].plot(kind='hist')
plt.subplot(2,2,3)
BSdata['体重'].plot(kind='box')
plt.subplot(2,2,4)
BSdata['体重'].plot(kind='density',title='Density')
plt.show()

#定性数据
T1 = BSdata['开设'].value_counts();T1
pd.DataFrame({'频数':T1,'频率':T1.T1.sum()*100})
T1.plot(kind='bar')
T1.plot(kind='pie')

f = BSdata['开设'].value_counts()
sum(f)
BSdata['开设'].value_counts(normalize=True)


def tab(x,plot=False):
    f = x.value_counts();f
    s = sum(f)
    p = round(f/s*100,3);p
    T1 = pd.concat([f,p],axis=1)
    T1.columns = ['例数','构成比']
    T2 = pd.DataFrame({'例数':s,'构成比':100.00},index=['合计'])
    Tab = T1.append(T2)
    if plot:
        fig,ax = plt.subplots(2,1,figsize=(8,15))
        ax[0].bar(f.index,f)
        ax[1].pie(p,labels=p.index,autopct='%1.2f%%')
    return(round(Tab,3))

def tab(x,plot=False): #计数频数表
    f=x.value_counts();f
    s=sum(f)
    p=round(f/s*100,3);p
    T1=pd.concat([f,p],axis=1)
    T1.columns=['例数','构成比']
    T2=pd.DataFrame({'例数':s,'构成比':100.00},index=['合计'])
    Tab=T1.append(T2)
    if plot:
        fig,ax = plt.subplots(2,1,figsize=(8,15))
        ax[0].bar(f.index,f); # 条图
        ax[1].pie(p,labels=p.index,autopct='%1.2f%%');
    return(round(Tab,3))

tab(BSdata.开设,True)  #Q 报错 显示函数未定义？？？

pd.cut(BSdata.身高,bins=10).value_counts()
pd.cut(BSdata.身高,bins=10).value_counts().plot(kind='bar')
plt.show()

pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts()

pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts().plot(kind='bar')
plt.show()

def freq(X,bins=10):
    H = plt.hist(X,bins);
    a = H[1][:-1];a
    b = H[1][1:];b
    f = H[0];f
    p = f/sum(f)*100;p
    cp = np.cumsum(p);cp
    Freq = pd.DataFrame([a,b,f,p,cp])
    Freq.index = ['下限','上限','频数','频率','累计频数（%）']
    return(round(Freq.T,2))

freq(BSdata.体重)
X = BSdata.体重

pd.crosstab(BSdata.开设,BSdata.课程)
pd.crosstab(BSdata.开设,BSdata.课程,margins=True)

pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='index')
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='columns')
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='all').round(3)

T2 = pd.crosstab(BSdata.开设,BSdata.课程);T2
T2.plot(kind='bar')
T2.plot(kind='bar',stacked=True)

BSdata.groupby(['性别'])
type(BSdata.groupby(['性别']))
BSdata.groupby(['性别'])['身高'].mean()
BSdata.groupby(['性别'])['身高'].size()
BSdata.groupby(['性别','开设'])['身高'].mean()

BSdata.groupby(['性别'])['身高'].agg([np.mean,np.std])

BSdata.groupby(['性别'])['身高','体重'].apply(np.mean)
BSdata.groupby(['性别','开设'])['身高','体重'].apply(np.mean)

BSdata.pivot_table(index=['性别'],values=['学号'],aggfunc=len)
BSdata.pivot_table(values=['学号'],index=['性别','开设'],aggfunc=len)
BSdata.pivot_table(values=['学号'],index=['开设'],columns=['性别'],aggfunc=len)

BSdata.pivot_table(index=['性别'],values=['身高'],aggfunc=np.mean)
BSdata.pivot_table(index=['性别'],values=['身高'],aggfunc=[np.mean,np.std])
BSdata.pivot_table(index=['性别'],values=['身高','体重'])

BSdata.pivot_table('学号',['性别','开设'],'课程',aggfunc=len,margins=True,margins_name='合计')
BSdata.pivot_table(['身高','体重'],['性别','开设'],aggfunc=[len,np.mean,np.std])

BSdata.学号

#05
#初等函数图

import math 
import numpy as np 
import matplotlib.pyplot as plt
x = np.linspace(0,2*math.pi);x
fig,ax = plt.subplots(2,2,figsize=(15,12))
ax[0,0].plot(x,np.sin(x))
ax[0,1].plot(x,np.cos(x))
ax[1,0].plot(x,np.log(x))
ax[1,1].plot(x,np.exp(x))
plt.show()

#极坐标图
t = np.linspace(0,2*math.pi)
x = 2*np.sin(t)
y = 3*np.cos(t)
plt.plot(x,y)
plt.text(0,0,r'$\frac{x^2}{2}+\frac{y^2}{3}=1$',fontsize=15)#fontsize字体大小
plt.show()

#气泡图
import pandas as pd 
BSdata = pd.read_csv('./BSdata.csv',encoding='utf-8')
plt.scatter(BSdata['身高'],BSdata['体重'],s=BSdata['支出'])

#三维曲面图
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X = np.linspace(-4,4,20) 
Y = np.linspace(-4,4,20)
X,Y = np.meshgrid(X,Y)
Z = np.sqrt(X**2+Y**2)
ax.plot_surface(X,Y,Z)
plt.show()

#三维散点图
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(BSdata['身高'],BSdata['体重'],BSdata['支出'])
plt.show()

import seaborn as sns 

sns.boxplot(x=BSdata['身高'])#横着放的箱线图
sns.boxplot(y=BSdata['身高'])#竖着放的箱线图
sns.boxplot(x='性别',y='身高',data=BSdata)#x为分组因子
sns.boxplot(x='开设',y='支出',hue='性别',data=BSdata)

sns.violinplot(x='性别',y='身高',data=BSdata)
sns.violinplot(x='开设',y='支出',hue='性别',data=BSdata)

sns.stripplot(x='性别',y='身高',data=BSdata)
sns.stripplot(x='性别',y='身高',data=BSdata,jitter=True)
sns.stripplot(y='性别',x='身高',data=BSdata,jitter=True)

sns.barplot(x='性别',y='身高',data=BSdata,ci=0,palette='Blues_d')

sns.countplot(x='性别',data=BSdata)
sns.countplot(y='开设',data=BSdata)
sns.countplot(x='性别',hue='开设',data=BSdata)

sns.catplot(x='性别',col='开设',col_wrap=3,data=BSdata,kind='count',height=2.5,aspect=0.8)

sns.distplot(BSdata['身高'],kde=True,bins=20,rug=True)
sns.jointplot(x='身高',y='体重',data=BSdata)
sns.pairplot(BSdata[['身高','体重','支出']])

from ggplot import *
import matplotlib.pyplot as plt #基本绘图包
plt.rcParams['font.sans-serif']=['KaiTi'] #SimHei黑体
plt.rcParams['axes.unicode_minus']=False #正常显示图中负号

qplot('身高',data=BSdata,geom='histogram')
qplot('开设',data=BSdata,geom='bar')

#散点图
qplot('身高','体重',data=BSdata,color='性别')

GP = ggplot(aes(x='身高',y='体重'),data=BSdata);GP

#直方图
ggplot(BSdata,aes(x='身高'))+ geom_histogram()

#散点图
ggplot(BSdata,aes(x='身高',y='体重'))+ geom_point()

#线图
ggplot(BSdata,aes(x='支出'))+geom_line(aes(y='身高'))

#分面图
ggplot(BSdata,aes(x='身高',y='体重'))+ geom_point() + facet_wrap('性别')

#添加主题
ggplot(BSdata,aes(x='身高',y='体重',color='性别'))+ geom_point() + theme_bw()