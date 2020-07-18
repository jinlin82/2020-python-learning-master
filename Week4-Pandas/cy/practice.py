import numpy as np

dat = np.random.randint(0,100,size=(50,8))

dat.iloc[1] #wrong
dat[1,3]
dat[[1,3,5],3:7]
dat[[1,3,5],:7]

x = [1,4,9,12,39,14]
x[2:5]
x[[1,4]] #wrong
np.array(x)[[1,4]]
###csv
import pandas as pd
import numpy as np 

# df is a pre-written file
df.to_csv('foo.csv') #save as csv
pd.read_csv('foo.csv') #read

### excel
df.to_excel('foo.xlsx',sheet_name='Sheet1')
pd.read_excel('foo.xlsx','Sheet1',index_col=None)

### Object Creation
##series
import pandas as pd
import numpy as np 
pd.Series([1,2,5,np.nan,8])

##DataFrame
dates = pd.date_range('20130101',periods=6)
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))

# DataFrame can be converted to series-like

df2 = pd.DataFrame({'A':1,'B':pd.Timestamp('20130102'),'C':pd.Series(1,index=list(range(4)),dtype='float32'),'D':np.array([3]*4,dtype='int32')})

# viewing Data
df.head()
df.tail(3)
df.index 
df.columns
df.values
df.describe() # a quick statistic summary of your data
df.T 

### sort

#Sorting by an axis
df.sort_index(axis=1,ascending=False)
#axis=1按列，ascending=False不按升序，即降序

#Soring by values
df.sort_values(by='B')

# selecting a single column
df['A'] # = df.A
#选择连续行
df[2:4]

#使用标签（label）选取:.loc[]
# 不能使用数字！！！

df2 = pd.DataFrame(np.random.randn(5,4),columns=list('ABCD'),index=pd.date_range('20130101',periods=5))
df2.loc[2:3] ## 错误
df2.loc['20130102':'20130104']
df2.loc[:,['A','C']]

#使用位置选取: .iloc[]
df2 = pd.DataFrame(np.random.randn(5,4),columns=list('ABCD'),index=pd.date_range('20130101',periods=5))
df2.iloc[2] # 取第3行的所有值
df2.iloc[2,3] #取（3，4）的那个值
df2.iloc[[0,2,4]] #第1，3，5行，带行名列名
df2.iloc[[0,2,4],:3] #上述的前3列

# .at .iat

#
df[df.A>0]
df[df>0]
df2.loc[df2['A']>0,'A':'C']
df2.iloc[list(df2['A']>0),0:3]

# isin
s = pd.Series(np.arange(5),index=np.arange(5)[::-1],dtype='int64')
s[s.isin([2,4,6])]

#
df = pd.DataFrame({'vals':[1,2,3,4],'ids':['a','b','f','n'],'ids2':['a','n','c','n']})
values = ['a','b',1,3]
df.isin(values)

values = {'ids':['a','b'],'vals':[1,3]}
df.isin(values)

#
values = {'ids':['a','b'],'ids2':['a','c'],'vals':[1,3]}
row_mask = df.isin(values).all(1)
df[row_mask]

#where
s[s>0]
s.where[s>0]
df.where(df<0,-df)
s.where(s<0,-s)

#Duplicate Data

df2 = pd.DataFrame({'a':['one','one','two','two','two','three','four'],'b':['x','y','x','y','x','x','x'],'c':np.random.randn(7)})
df2.duplicated('a')
df2.duplicated('a',keep='last')
df2.duplicated('a',keep=False)
df2.drop_duplicates('a')
df2.drop_duplicates('a',keep='last')
df2.drop_duplicates('a',keep=False)

df2.duplicated(['a','b'])
df2.drop_duplicates(['a','b'])

#statistic
df.mean() #总的
df.mean(1) #第一行
np.mean(np.array(df)) #报错

#apply
df.apply(np.cumsum) #按行加总
df.apply(lambda x:x.max()-x.min())#报错

# Series
s = pd.Series(['A','B','C','AaBa','Baca',np.nan,'CABA','dog','cat'])
s.str.lower() # 转换为小写

#concat
df = pd.DataFrame(np.random.randn(10,4))
pieces = [df[:3],df[3:7],df[7:]]
pd.concat(pieces)

#append
df = pd.DataFrame(np.random.randn(8,4),columns=['A','B','C','D'])
s = df.iloc[3]
df.append(s) #df后面加上s
df.append(s,ignore_index=True) #忽略标签s

#
df = pd.DataFrame({'A':['foo','bar','foo','bar','foo','bar','foo','foo'],'B':['one','one','two','three','two','two','one','three'],'C':np.random.randn(8),'D':np.random.randn(8)})
df.groupby('A').sum()
df.groupby(['A','B']).sum()

df.groupby(['A','B']).agg([np.sum,np.mean,np.std])
groupby.agg({'C':np.sum,'D':lambda x:np.std(x,ddof=1)})

#plot
import matplotlib.pyplot as plt 
np.random.seed(1234)
df = pd.DataFrame(np.random.randn(50,2))
df['g'] = np.random.choice(['A','B'],size=50)
df.loc[df['g']=='B',1] += 3
df.groupby('g').boxplot()

# 使用pivot()方法
import pandas.util.testing as tm;tm.N = 3
def unpivot(frame):
    N,K = frame.shape
    date = {'value':frame.values.ravel('F'),'variable':np.asarray(frame.columns).repeat(N),'date':np.tile(np.asarray(frame.index),K)}
    return pd.DataFrame(date,columns=['date','variable','value'])

df = unpivot(tm.makeTimeDataFrame())
df[df['variable'] == 'A']
df.pivot(index='date',columns='variable',values='value')
df['value2'] = df['value']*2
df.pivot('date','variable')

pivoted = df.pivot('date','variable')
pivoted['value2']

#cross tabulations
foo,bar,dull,shiny,one,two = 'foo','bar','dull','shiny','one','two'
a = np.array([foo,foo,bar,bar,foo,foo],dtype=object)
b = np.array([one,one,two,one,two,one],dtype=object)
c = np.array([dull,dull,shiny,dull,dull,shiny],dtype=object)
pd.crosstab(a,[b,c],rownames=['a'],colnames=['b','c'])
df = pd.DataFrame({'A':[1,2,2,2,2],'B':[3,3,4,4,4],'C':[1,1,np.nan,1,1]})
pd.crosstab(df.A,df.B)
pd.crosstab(df['A'],df['B'],normalize=True)
pd.crosstab(df['A'],df['B'],normalize='columns')
pd.crosstab(df.A,df.B,values=df.C,aggfunc=np.sum,normalize=True,margins=True)

#bin
ages = np.array([10,15,13,12,23,25,28,59,60])
pd.cut(ages,bins=3)
pd.cut(ages,bins=[0,18,35,70])

#dummy variables
df = pd.DataFrame({'key':list('bbacab'),'data1':range(6)})
pd.get_dummies(df['key'])
dummies = pd.get_dummies(df['key'],prefix='key')
df = pd.DataFrame({'A':['a','b','a'],'B':['c','c','b'],'C':[1,2,3]})
pd.get_dummies(df['key'])
pd.get_dummies(df,columns=['A'])

S = pd.Series(['a','b','c','a'],dtype='category')
df = pd.DataFrame({'id':[1,2,3,4,5,6],'raw_grade':['a','b','b','a','a','e']})
df['grade'] = df['raw_grade'].astype('category')
df['grade'].cat.categories = ['very good','good','very bad']
df['grade'] = df['grade'].cat.set_categories(['very bad','bad','medium','good','very good'])
df.sort_values(by='grade')
df.groupby('grade').size()

#plot
ts = pd.Series(np.random.randn(1000),index=pd.date_range('1/1/2000',periods=1000))
ts = ts.cumsum()
ts.plot()
df3 = pd.DataFrame(np.random.randn(1000,2),columns=['B','C']).cumsum()
df3['A'] = pd.Series(list(range(len(df3))))
df3.plot(x='A',y='B')

##
import numpy as np
import pandas as pd
BSdata = pd.read_csv('./BSdata/BSdata.csv')
BSdata.describe()
BSdata[['性别','开设','课程','软件']].describe()
#定性数据
T1 = BSdata.性别.value_counts();T1
T1/sum(T1)*100

BSdata.身高.mean()
BSdata.身高.max() - BSdata.身高.min()
BSdata.身高.var()
BSdata.身高.std()
BSdata.身高.quantile(0.75) - BSdata.身高.quantile(0.25)
BSdata.身高.skew() #偏度
BSdata.身高.kurt() #峰度

#

def stats(x):
    stat = [x.count(),x.min(),x.quantile(.25),x.mean(),x.median(),x.quantile(.75),x.max(),x.max()-x.min(),x.var(),x.std(),x.skew(),x.kurt()]
    stat = pd.Series(stat,index=['Count','Min','Q1(25%)','Mean','Median','Q3(75%)','Max','Range','Var','Std','Skew','Kurt'])
    return(stat)
stats(BSdata.身高)
stats(BSdata.支出)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['KaiTi'] #SimHei黑体
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(6,5))

X = ['A','B','C','D','E','F','G']
Y = [1,4,7,3,2,5,6]
plt.bar(X,Y)
plt.savefig('abc',format = 'pdf')

plt.pie(Y,labels=X)
plt.show()

plt.plot(X,Y) #线图

plt.hist(BSdata.身高) #频数直方图
plt.hist(BSdata.身高,density=True) #频率直方图

plt.scatter(BSdata.身高,BSdata.体重)

plt.ylim(0,8)
plt.xlabel('name');plt.ylabel('values')
plt.xticks(range(len(X)),X)
plt.plot(X,Y,linestyle='--',marker='o')

plt.plot(X,Y,'o--')
plt.axvline(x=1)#纵坐标y处画垂直线（plt.axvline）
plt.axhline(y=4)#横坐标x处画水平线（plt.axhline）

plt.plot(X,Y)
plt.text(2,7,'peakpoint') #终端显示不了
plt.show()

plt.plot(X,Y,label=u'折线')
plt.legend()
plt.show()

s = [0.1,0.4,0.7,0.3,0.2,0.5,0.6]
plt.bar(X,Y,yerr=s,error_kw={'capsize':5})

plt.figure(figsize=(12,6))
plt.subplot(1,2,1);plt.bar(X,Y)
plt.subplot(1,2,2);plt.plot(Y)
plt.show()

fig,ax = plt.subplots(1,2,figsize=(14,6))
ax[0].bar(X,Y)
ax[1].plot(X,Y)
plt.show()

#一页绘制四个图形
fig,ax = plt.subplots(2,2,figsize=(15,10))
ax[0,0].bar(X,Y);ax[0,1].pie(Y,labels = X)
ax[1,0].plot(Y);ax[1,1].plot(Y,'.-',linewidth=3)
plt.show()

BSdata['体重'].plot(kind='line')
BSdata['体重'].plot(kind='hist')

BSdata['体重'].plot(kind='line')
BSdata['体重'].plot(kind='hist')
BSdata['体重'].plot(kind='box')
BSdata['体重'].plot(kind='density',title='Density')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(1,3),kind='box')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(1,3),kind='density')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(3,1),kind='density')

T1 = BSdata['开设'].value_counts();T1
pd.DataFrame({'频数':T1,'频率':T1/T1.sum()*100})
T1.plot(kind='bar'); #T1.sort_values().plot(kind='bar')
T1.plot(kind='pie')

#pivot-table
BSdata['开设'].value_counts()
BSdata.pivot_table(values='学号',index='开设',aggfunc=len)

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
tab(BSdata.开设,True)

#定量数据
pd.cut(BSdata.身高,bins=10).value_counts()
pd.cut(BSdata.身高,bins=10).value_counts().plot(kind='bar')

#支出频数表
pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts()
pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts().plot(kind='bar')

def freq(X,bins=10):#计量频数表与直方图
    H = plt.hist(X,bins)
    a = H[1][:-1];a #去尾
    b = H[1][1:];b #去头
    f = H[0];f
    p = f/sum(f)*100;p
    cp = np.cumsum(p);cp
    Freq = pd.DataFrame([a,b,f,p,cp])
    Freq.index = ['下限','上限','频数','频率（%）','累计频数（%）']
    return(round(Freq.T,2))
freq(BSdata.体重)
X = BSdata.体重

#二维列联表
pd.crosstab(BSdata.开设,BSdata.课程)

#行和列的合计
pd.crosstab(BSdata.开设,BSdata.课程,margins=True)

#边缘概率
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='index')
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='columns')
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='all').round(3)

#复式条图
T2 = pd.crosstab(BSdata.开设,BSdata.课程);T2
T2.plot(kind='bar')
T2.plot(kind='bar',stacked=True)

#按列分组
BSdata.groupby(['性别'])
type(BSdata.groupby(['性别']))

BSdata.groupby(['性别'])['身高'].mean()
BSdata.groupby(['性别'])['身高'].size()
BSdata.groupby(['性别','开设'])['身高'].mean()

#应用agg(func) agg 用于DataFrame的各个列
BSdata.groupby(['性别'])['身高'].agg([np.mean,np.std])

#apply仅作用于指定的列
BSdata.groupby(['性别'])['身高','体重'].apply(np.mean)
BSdata.groupby(['性别','开设'])['身高','体重'].apply(np.mean)

#定性数据透视分析 pivot-table
BSdata.pivot_table(index=['性别'],values=['学号'],aggfunc=len)
BSdata.pivot_table(values=['学号'],index=['性别','开设'],aggfunc=len)
BSdata.pivot_table(values=['学号'],index=['开设'],columns=['性别'],aggfunc=len)

#
BSdata.pivot_table(index=['性别'],values=['身高'],aggfunc=np.mean)
BSdata.pivot_table(index=['性别'],values=['身高'],aggfunc=[np.mean,np.std])
BSdata.pivot_table(index=['性别'],values=['身高','体重'])

#复合数据透视分析
BSdata.pivot_table('学号',['性别','开设'],'课程',aggfunc=len,margins=True,margins_name='合计')
BSdata.pivot_table(['身高','体重'],['性别','开设'],aggfunc=[len,np.mean,np.std])


import numpy as np
import pandas as pd 
import os
#path = './data/'
#workdata = pd.read_excel('../data/数据.xls',0)
#def get_file():
#    files = os.listdir(path)
#    files.sort()
#    list = []
#    for file in files:
#        if not os.path.isdir(path+file):
#            f_name = str(file)
#            filename = path + f_name
#            list.append(filename)
#    return list
#list = get_file()
#datas = []
#for i in range(len(list)):
#    data = pd.read_csv(list[i],encoding = 'gbk')
#    datas.append(data)
#return(datas)
#data = pd.read_csv(list[2],encoding = 'gbk')
gdp = pd.read_csv('./data/gdp.csv',encoding = 'gb2312')
gdp.iloc[:,1:].T.values.reshape(-1)
csvfiles = os.listdir('./data')

dat = pd.DataFrame()
for i in csvfiles:
    temp = pd.read_csv('./data/'+i,encoding='gb2312')
    temp1 = temp.iloc[:,1:].T.values.reshape(180)
    dat[i[:-4]] = temp1

year = np.arange(2002,2012)
#np.repeat(year,18)
dis = temp.district
#np.repeat(dis.values.reshape((18,1)),10,axis=1).T.reshape(-1)
#np.tile(dis,10)
#type(dis)
dat['Year'] = np.repeat(year,18)
dat['dis'] = np.tile(dis,10)
dat['dis'] = list(dis)*10

Year = pd.Series(np.repeat(year,18))
District = pd.Series(np.tile(dis,10))
pd.concat((np.repeat(year,18),np.tile(dis,10),dat))

dat = pd.concat((Year,District,dat),axis=1)
dat.columns.values[:2] = ['Year','District']


dat.to_csv('./dat.csv',encoding='gb2312') #写入当前文件夹

Year,District = np.meshgrid(dis,year)
dat = pd.concat((pd.Series(Year.reshape(-1)),pd.Series(District.reshape(-1)),dat),axis=1)
#dat.columns.values[:2] = ['Year','District']

##斐波纳契函数：后面一个元素等于前面俩元素之和

x = 100
x1 = 1
x2 = 2
x3 = []
x3.append(x1+x2)

X1 = x2
X2 = x3
X3 = X1+X2

while x3[-1]<=10000000:
    x1 = x2
    x2 = x3[-1]
    x3.append(x1+x2)

def fibo(x):
    x1 = x2 =1
    x3 = [1,1]
    while x3[-1]<=x:
        x1 = x2
        x2 = x3[-1]
        x3.append(x1+x2)
    return x3[:-1]
fibo(100)

Year = pd.DataFrame(np.repeat(year,18),columns=['Year'])
District = pd.DataFrame(np.tile(dis,10),columns=['Dis'])