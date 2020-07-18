### 王斌会-第三章 ###
import pandas as pd 
pd.Series()

x=[1,3,6,4,9]
s1=pd.Series(x);s1
y=[1,3,5,7,9]
s2=pd.Series(y);s2

pd.concat([s1,s2],axis=0)
pd.concat([s1,s2],axis=1)

s1[2]
s1[1:3]

pd.DataFrame(x)
pd.DataFrame(x,columns=['x'],index=range(5))  #给行列标题命名

df=pd.DataFrame({'x':x,'y':y});df

df['y2']=df['y']**2;df  
#两个*表示平方

del df['y2'];df

df.isnull()
df.isnull().sum()

df.sort_index()
df.sort_values(by='x')

###for循环语句
#####注意冒号：/空四格/全选运行
for i in range(1,5):
    print(i)

fruits=['banana','apple','mango']
for fruit in fruits:
    print('当前水果：',fruit)

###条件语句if/else 
a=-100
if a<100:
    print('数值小于100')
else:
    print('数值大于100')

#函数定义
x=[1,3,6,4,9,7,5,8,2];x
def xbar(x):
    n=len(x)
    xm=sum(x)/n 
    return(xm)
xbar(x)

sum(x)
len(x)

import numpy as np 
x=np.array([1,3,6,4,9,7,5,8,2]);x
def ss1(x):
    n=len(x)
    ss=sum(x**2)-sum(x)**2/n 
    return(ss)
ss1(x)

def ss2(x):
    n=len(x)
    xm=sum(x)/n 
    ss=sum(x**2)-sum(x)**2/n 
    return(x**2,n,xm,ss)
ss2(x)


#### 王斌会-第四章 ####
import matplotlib.pyplot as plt 
plt.rcParams['font.sans-scrif']=['KaiTi'] #错误？
plt.rcParams['axes.unicode minus']=False  #错误？
plt.figure(figsize=(5,4))

x=['a','b','c','d','e','f','g']
y=[1,4,7,3,2,5,6]
plt.bar(x,y)   #条形图
plt.pie(y,labels=x)  #饼图
plt.plot(x,y)  #线图

#图形参数设置
plt.plot(x,y,c='red');plt.ylim(0,8);
plt.xlabel('names');plt.ylabel('values');
plt.xticks(range(len(x)),x)

plt.plot(x,y,linestyle='--',marker='o')  #虚线；实心圆
##添加垂直线
plt.plot(x,y,'o--');plt.axvline(x=1);plt.axhline(y=4)

plt.plot(x,y);plt.text(2,7,'pack point')   #指定点添加文字
plt.plot(x,y,label='折线');plt.legend()  #图例

s=[0.1,0.4,0.7,0.3,0.2,0.5,0.6]
plt.bar(x,y,yerr=s,crror_kw={'capsize':5})  #???误差条图

##多个子图
#从左到右、从上到下对子区域编号
plt.subplot(121);plt.bar(x,y)
plt.subplot(122);plt.plot(y)  #一行

plt.subplot(211);plt.bar(x,y)
plt.subplot(212);plt.plot(y)  #一列

fig,ax=plt.subplot(1,2,figsize=(15,6))###???报错
ax[0].bar(x,y)
ax[1].plot(x,y)    #一页

### 网课视频代码 7-8 ###
##### 对应教材：03.beamer-python编程

# 条件语句if/else:
a=-100
if a<100:
    print('数值小于100')    #注意要空4格，按Tab快捷键,空格决定语句归属于谁
else:
    print('数值大于100')

if a>100:
    print('数值大于100')
print(a)

a=500
if(a>100){
    print('数值大于100')
    print(a)
}                 #也可以用花括号来写

if a>100:
    print('数值大于100')
else:
    print('数值小于等于100')

print('数值大于100') if a>100 else print(a)
#当if成立时只做一条语句结果，可以这样简写

### 循环语句 for...in...:

for i in range(1,11):
    print(i)

import numpy as np
np.arange(1,11)

#输出26个字母
import string
for i in string.ascii_lowercase:
    print(i)

x=np.random.randint(-10,10,15);x 
for i in x:
    if i>=0:
        print(i**0.5)
    else:
        break     #循环终止：break

for i in x:
    if i>=0:
        print(i**0.5)
    else:
        continue   #循环继续：continue


fruits=['banana','apple','mango']
for fruit in fruits:
    print('当前水果：',fruit)


#### 自定义函数
y=[1,8,3,19,20,-4,5,2,8,47]
type(y)
sum(y)
len(y)

#定义均值函数mean():
def mean(Lst):
    return sum(Lst)/len(Lst)

mean(y)


#将某个列表放入一个空列表
y=[1,8,3,19,20,-4,5,2,8,47]
y1=list()
for i in y:
    y1.append(i-1)   #appen()用于在列表末尾添加新的对象
y1

##先定义离差平方列表ssq():
def ssq(Lst,x):
    res=list()
    for i in Lst:
        res.append((i-x)**2)
    return res 

#再定义样本方差函数var():
def var(Lst):
    Lst_mean=mean(Lst)
    return sum(ssq(Lst,Lst_mean))/(len(Lst)-1)

var(y) 
#检验
np.var(y,ddof=1)   #ddof=1表示求样本方差，np.var()默认是总体方差

type(ssq)

#### class 类
class Myclass:
    '''第一个类'''
    i=1234
    def f(self):
        print('hello world')

a=Myclass()
type(a)
dir(a)
a.i    #属性
a.f()

def var(Lst):
    '''计算样本方差'''
    Lst_mean=mean(Lst)
    return sum(ssq(Lst,Lst_mean))/(len(Lst)-1)
help(var)

class circle:
    '''一个二维平面中的圆'''
    pos=(0,0)
    r=1

#在某个类里增加方法method
class circle:
    '''一个二维平面中的圆'''
    def __init__(self,t=(0,0),x=1):   #默认设置
        self.pos=t
        self.r=x
    def __str__(self):
        print('圆:','位置为:',self.pos,'半径为:',self.r)


d=circle()
d.pos

g=circle((1,2),20)
g.pos

b=circle()
type(b)
dir(b)
b.pos
b.r

c=circle((1,1),10)
dir(c)
c.pos
c.r

#定义函数function
import numpy as np 
#定义圆的面积函数c_area():
def c_area(x):
    return np.pi*x.r**2

g=circle((1,2),20)
c_area(g)

class circle:
    '''一个二维平面中的圆'''
    def __init__(self,t=(0,0),x=1):   #默认设置
        self.pos=t
        self.r=x
    def __str__(self):
        print('圆:','位置为:',self.pos,'半径为:',self.r)
    def area(self):
        return np.pi*self.r**2
    def dis2(self,other):      #求两个圆的距离
        return ((self.pos[1]-other.pos[1])**2+(self.pos[0]-other.pos[0])**2)**0.5

h=circle((1,1),4)
dir(h)          #可发现新增method方法

k=circle((3,8),3)
h.dis2(k)    #得到h和k的圆心距


##### 04_beamer.pdf #####
import numpy as np
import pandas as pd 
BSdata=pd.read_csv('./data/BSdata.csv')
BSdata.head()
BSdata.describe()    #结果只包含数值型数据的描述
BSdata[['性别','开设','课程','软件']].describe()   #定性数据描述

### 定型数据汇总分析
T1=BSdata.性别.value_counts();T1    #频数
T1/sum(T1)*100        #频率

### 定量数据汇总分析
BSdata.身高.mean()
BSdata.身高.median()
BSdata.身高.max()-BSdata.身高.min()
BSdata.身高.var()
BSdata.身高.std()
BSdata.身高.quantile(0.75)-BSdata.身高.quantile(0.25)
BSdata.身高.skew()
BSdata.身高.kurt()


### 自编计算基本统计量函数
def stats(x):
    stat=[x.count(),x.min(),x.quantile(.25),x.mean(),x.median(),x.quantile(.75),x.max()-x.min(),x.var(),x.std(),x.skew(),x.kurt()]
    stat=pd.Series(stat,index=['Count','Min', 'Q1(25%)','Mean','Median','Q3(75%)','Max','Range','Var','Std','Skew','Kurt' ])

stats(BSdata.身高)   #运行错误
stats(BSdata.支出)

### 常用的绘图函数
import matplotlib.pyplot as plt          #基本绘图包
plt.rcParams['font.sans-serif']=['KaiTi']     #SimHei黑体
plt.rcParams['axes.unicode_minus']=False     #正常显示图中负号
# plt.figure(figsize=(6,5))     #图形大小

### 常见的统计作图函数
X=['A','B','C','D','E','F','G']
Y=[1,4,7,3,2,5,6]
plt.bar(X,Y)    #条图
plt.savefig('abc',format='pdf')

plt.pie(Y,labels=X)   #饼图
plt.show()      #两行选中同时运行

plt.plot(X,Y)    #线图

plt.hist(BSdata.身高)      #频数直方图
plt.hist(BSdata.身高,density=True)   #频率直方图

plt.scatter(BSdata.身高,BSdata.体重)   #散点图

### 标题、标签、标尺及颜色
plt.ylim(0,8);
plt.xlabel('names');plt.ylabel('values');
plt.xticks(range(len(X)),X)

### 线型和符号
plt.plot(X,Y,linestyle='--',marker='o')

plt.plot(X,Y,'o--')
plt.axvline(x=1)
plt.axhline(y=4)

plt.plot(X,Y)
plt.text(2,7,'peakpoint')   #在（x,y）处添加用 labels 指定的文字

###  legend 函数给图形加图例
plt.plot(X,Y,label=u'折线')
plt.legend()                 #图例

s=[0.1,0.4,0.7,0.3,0.2,0.5,0.6]
plt.bar(X,Y,yerr=s,error_kw={'capsize':5})  #误差条图

### 多图

#'''一行绘制两个图形'''
plt.figure(figsize=(12,6))
plt.subplot(1,2,1); plt.bar(X,Y)
plt.subplot(1,2,2); plt.plot(Y)     #一行

 #'''一列绘制两个图形'''
plt.figure(figsize=(7,10))
plt.subplot(2,1,1); plt.bar(X,Y)
plt.subplot(2,1,2); plt.plot(Y)     #一列

 #'''一页绘制两个图形'''
fig,ax = plt.subplots(1,2,figsize=(14,6))
ax[0].bar(X,Y)
ax[1].plot(X,Y)                     #一页

 #'''一页绘制四个图形'''
fig,ax=plt.subplots(2,2,figsize=(15,10))
ax[0,0].bar(X,Y); ax[0,1].pie(Y,labels=X)
ax[1,0].plot(Y); ax[1,1].plot(Y,'.-',linewidth=3)

### 基于pandas的绘图 （kind=...）
DataFrame.plot(kind='line')

BSdata['体重'].plot(kind='line')
BSdata['体重'].plot(kind='hist')
BSdata['体重'].plot(kind='box')
BSdata['体重'].plot(kind='density',title='Density')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(1,3),kind='box')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(1,3),kind='density')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(3,1),kind='density')

T1=BSdata['开设'].value_counts();T1
pd.DataFrame({'频数':T1,'频率':T1/T1.sum()*100})
T1.plot(kind='bar'); #T1.sort_values().plot(kind='bar')
T1.plot(kind='pie')

### 定性数据的频数分布
BSdata['开设'].value_counts()
#BSdata.pivot_table(values='学号',index='开设',aggfunc=len)

def tab(x,plot=False): #计数频数表
    f=x.value_counts();f
    s=sum(f);
    p=round(f/s*100,3);p
    T1=pd.concat([f,p],axis=1);
    T1.columns=['例数','构成比'];
    T2=pd.DataFrame({'例数':s,'构成比':100.00},index=['合计'])
    Tab=T1.append(T2)
    if plot:       #plot=True
        fig,ax = plt.subplots(2,1,figsize=(8,15))
        ax[0].bar(f.index,f); # 条图
        ax[1].pie(p,labels=p.index,autopct='%1.2f%%'); # 饼图
        return(round(Tab,3))

tab(BSdata.开设,True)

pd.cut(BSdata.身高,bins=10).value_counts()
pd.cut(BSdata.身高,bins=10).value_counts().plot(kind='bar')

pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts()
pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts().plot(kind='bar')       #注意bins=[]的意义，划分组别


def freq(X,bins=10): #计量频数表与直方图
    H=plt.hist(X,bins);
    a=H[1][:-1];a
    b=H[1][1:];b
    f=H[0];f
    p=f/sum(f)*100;p
    cp=np.cumsum(p);cp
    Freq=pd.DataFrame([a,b,f,p,cp])
    Freq.index=['[下限','上限)','频数','频率(%)','累计频数(%)']
    return(round(Freq.T,2))
    
freq(BSdata.体重)

### 二维聚集分析  crosstab() 函数
pd.crosstab(BSdata.开设,BSdata.课程)    #二维
pd.crosstab(BSdata.开设,BSdata.课程,margins=True)   #合计

pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='index')   #占行的比例
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='columns')   #占列的比例
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='all').round(3)   #占总和的比例


T2=pd.crosstab(BSdata.开设,BSdata.课程);T2
T2.plot(kind='bar')           #复式条图；分段式

T2.plot(kind='bar',stacked=True)    #复试条图；并列式


### 定量数据的集聚表  groupby()函数
BSdata.groupby(['性别'])
type(BSdata.groupby(['性别']))

BSdata.groupby(['性别'])['身高'].mean()
BSdata.groupby(['性别'])['身高'].size()
BSdata.groupby(['性别','开设'])['身高'].mean()

BSdata.groupby(['性别'])['身高'].agg([np.mean, np.std])  #运用agg()函数

BSdata.groupby(['性别'])['身高','体重'].apply(np.mean)
BSdata.groupby(['性别','开设'])['身高','体重'].apply(np.mean)
## apply()不同于agg()的地方在于：前者应用于 dataFrame 的各个列，后者仅作用于指定的列


### 多维透视分析  pivot_table 
BSdata.pivot_table(index=['性别'],values=['学号'],aggfunc=len)
BSdata.pivot_table(index=['性别','开设'],values=['学号'],aggfunc=len)
BSdata.pivot_table(values=['学号'],index=['开设'],columns=['性别'],aggfunc=len)

BSdata.pivot_table(index=['性别'],values=['身高'],aggfunc=np.mean)
BSdata.pivot_table(index=['性别'],values=['身高'],aggfunc=[np.mean,np.std])
BSdata.pivot_table(index=['性别'],values=['身高','体重'])  #默认为均值

# 复合数据透视分析
BSdata.pivot_table('学号', ['性别','开设'], '课程', aggfunc=len, margins=True, margins_name= '合计')
BSdata.pivot_table(['身高','体重'],['性别','开设'],aggfunc=[len,np.mean,np.std])






