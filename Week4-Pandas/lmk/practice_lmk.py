# 控制语句
## 循环语句 for
for i in range(1,5):
    print(i)

fruits=['banana','apple','mango']
for fruit in fruits:
    print('当前水果：',fruit)

import pandas as pd
BSdata=pd.read_csv("./BSdata.csv",encoding='utf-8')
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
    stat=[x.count(),x.min(),x.quantile(0.25),x.mean(),x.median(),x.quantile(0.75),
          x.max(),x.max()-x.min(),x.var(),x.std(),x.skew(),x.kurt()]
    stat=pd.Series(stat,index=['Count','Min','Q1(25%)','Mean','Median','Q3(75%)',
                   'Max','Range','Var','Std','Skew','Kurt'])
    x.plot(kind='kde') # 拟合核密度曲线
    return(stat)
stats(BSdata.身高)
stats(BSdata.支出)

# 基本绘图命令
import matplotlib.pyplot as plt # 基本绘图包
plt.rcParams['font.sans-serif']=['KaiTi'] # KaiTi楷体/SimHei黑体/SimSun宋体
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
T1=BSdata['开设'].value_counts();T1
pd.DataFrame({'频数':T1,'频率':T1/T1.sum()*100})

T1.plot(kind='bar')
T1.sort_values().plot(kind='bar')
T1.plot(kind='pie')

# 一维频数分析
## 计数频数分析
BSdata['开设'].value_counts()
BSdata.pivot_table(values='学号',index='开设',aggfunc=len)

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
# 中文图例的显示需要设置字体

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

# pandas
## objects creation
import pandas as pd
import numpy as np
pd.Series([1,2,5,np.nan,8])

dates=pd.date_range('20130101',periods=6)
df=pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))

df2=pd.DataFrame({'A':1.,
                  'B':pd.Timestamp('20130102'),
                  'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                  'D' : np.array([3]*4,dtype='int32'),
                  'E' : pd.Categorical(["test","train","test","train"]),
                  'F' : 'foo' })

## viewing data
df.head()
df.tail(3)

df.index
df.columns
df.values

df.describe()

df.T

## sorting
df.sort_index(axis=1,ascending=False) # 降序
df.sort_values(by='B')

## selection
df['A']
df[2:4]

df2=pd.DataFrame(np.random.randn(5,4),columns=list('ABCD'),index=pd.date_range('20130101',periods=5))
df2.loc[2:3] # 错误
df2.loc['20130102':'20130104']
df2.loc[:,['A','C']]

df2.iloc[2]
df2.iloc[2,3]
df2.iloc[[0,2,4]]
df2.iloc[[0,2,4],:3]

### boolean indexing
df[df.A>0]
df[df>0]
df2.loc[df2['A']>0,'A':'C']
df2.iloc[list(df2['A']>0),0:3]

### indexing with isin
s=pd.Series(np.arange(5),index=np.arange(5)[::-1],dtype='int64')
s[s.isin([2,4,6])]

df=pd.DataFrame({'vals':[1,2,3,4],'ids':['a','b','f','n'],'ids2':['a','n','c','n']})
values=['a','b',1,3]
df.isin(values)

values={'ids':['a','b'],'vals':[1,3]}
df.isin(values)

values={'ids':['a','b'],'ids2':['a','c'],'vals':[1,3]}
row_mask=df.isin(values).all(1)
df[row_mask]

### the where() method
s[s>0]
s.where(s>0)
df.where(df<0,-df) # 报错，()里不要给str

## duplicate data
df2=pd.DataFrame({'a':['one','one','two','two','two','three','four'],
                  'b':['x','y','x','y','x','x','x'],
                  'c':np.random.randn(7)})
df2.duplicated('a')
df2.duplicated('a',keep='last')
df2.duplicated('a',keep=False)
df2.drop_duplicates('a')
df2.drop_duplicates('a',keep='last')
df2.drop_duplicates('a',keep=False)
# keep='first'(default): mark/drop duplicates except for the first occurrence.
# keep='last': mark/drop duplicates except for the last occurrence.
# keep=False: mark/drop all duplicates.
df2.duplicated(['a','b'])
df2.drop_duplicates(['a','b'])

## apply函数
df.apply(np.cumsum)
df.apply(lambda x:x.max()-x.min()) ###

## 字符处理
s=pd.Series(['A','B','C','Aaba','Baca',np.nan,'CABA','dog','cat'])
s.str.lower()

## merge
### concat
df=pd.DataFrame(np.random.randn(10,4))
pieces=[df[:3],df[3:7],df[7:]]
pd.concat(pieces)

### append
df=pd.DataFrame(np.random.randn(8,4),columns=['A','B','C','D'])
s=df.iloc[3]
df.append(s)
df.append(s,ignore_index=True)

## grouping
df=pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'],
                 'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                 'C': np.random.randn(8),
                 'D': np.random.randn(8)});df
df.groupby('A').sum()
df.groupby(['A','B']).sum()

## aggregation
df.groupby(['A','B']).agg([np.sum,np.mean,np.std])

grouped.agg({'C':np.sum,
             'D':lambda x: np.std(x,ddof=1)}) ###

import matplotlib.pyplot as plt
np.random.seed(1234)
df=pd.DataFrame(np.random.randn(50,2))
df['g']=np.random.choice(['A','B'],size=50)
df.loc[df['g']=='B',1]+=3
df.groupby('g').boxplot() ###

## reshaping
### pivot() function
import pandas.util.testing as tm;tm.N=3
def unpivot(frame):
    N,K=frame.shape
    data={'value':frame.values.ravel('F'),
          'variable':np.asarray(frame.columns).repeat(N),
          'date':np.tile(np.asarray(frame.index),K)}
    return pd.DataFrame(data,columns=['date','variable','value'])
df=unpivot(tm.makeTimeDataFrame())

df[df['variable']=='A']
df.pivot(index='date',columns='variable',values='value')
df['value2']=df['value']*2
df.pivot('date','variable')

pivoted=df.pivot('date','variable')
pivoted['value2']

### pivot tables
df=pd.DataFrame({'A':['foo','foo','foo','foo','foo','bar','bar','bar','bar'],
                 'B':['one','one','one','two','two','one','one','two','two'],
                 'C':['small','large','large','small','small','large','small','small','large'],
                 'D':[1, 2, 2, 3, 3, 4, 5, 6, 7],
                 'E':[2, 4, 5, 5, 6, 6, 8, 9, 9]})

table=pd.pivot_table(df,values='D',index=['A','B'],columns=['C'],aggfunc=np.sum)
table=pd.pivot_table(df,values=['D','E'],index=['A','C'],aggfunc={'D':np.mean,'E':np.mean})
table=pd.pivot_table(df,values=['D','E'],index=['A','C'],aggfunc={'D':np.mean,'E':[min,max,np.mean]})

### cross tabulations
foo,bar,dull,shiny,one,two='foo','bar','dull','shiny','one','two'
a=np.array([foo,foo,bar,bar,foo,foo],dtype=object)
b=np.array([one,one,two,one,two,one],dtype=object)
c=np.array([dull,dull,shiny,dull,dull,shiny],dtype=object)
pd.crosstab(a,[b,c],rownames=['a'],colnames=['b','c'])

df=pd.DataFrame({'A':[1,2,2,2,2],'B':[3,3,4,4,4],'C':[1,1,np.nan,1,1]})
pd.crosstab(df.A,df.B)
pd.crosstab(df['A'],df['B'],normalize=True)
pd.crosstab(df['A'],df['B'],normalize='columns')
pd.crosstab(df.A,df.B,values=df.C,aggfunc=np.sum,normalize=True,margins=True)

### cut function
ages=np.array([10,15,13,12,23,25,28,59,60])
pd.cut(ages,bins=3)
pd.cut(ages,bins=[0,18,35,70])

### get_dummies()
df=pd.DataFrame({'key':list('bbacab'),'data1':range(6)})
pd.get_dummies(df['key'])
dummies=pd.get_dummies(df['key'],prefix='key')

df=pd.DataFrame({'A':['a','b','a'],'B':['c','c','b'],'C':[1,2,3]})
pd.get_dummies(df,columns=['A'])

## categoricals
s=pd.Series(['a','b','c','a'],dtype='category')
df=pd.DataFrame({'id':[1,2,3,4,5,6],'raw_grade':['a','b','b','a','a','e']})
df['grade']=df['raw_grade'].astype('category')
df['grade'].cat.categories=['very good','good','very bad']
df['grade']=df['grade'].cat.set_categories(['very bad','bad','medium','good','very good'])

df.sort_values(by='grade')
df.groupby('grade').size()

### plot()
ts=pd.Series(np.random.randn(1000),index=pd.date_range('1/1/2000',periods=1000))
ts=ts.cumsum()
ts.plot()

df3=pd.DataFrame(np.random.randn(1000,2),columns=['B','C']).cumsum()
df3['A']=pd.Series(list(range(len(df))))
df3.plot(x='A',y='B')