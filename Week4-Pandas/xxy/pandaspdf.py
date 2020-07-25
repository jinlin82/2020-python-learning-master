import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
df.to_csv('./data/BSdata.csv')

pd.read_csv('./data/BSdata.csv')

pd.Series([1,2,5,np.nan,8])

dates=pd.date_range('20130101',periods=6)
df=pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
df=pd.DataFrame(np.random.randn(6,4),index=dates,columns=['A','B','C','D'])

df2=pd.DataFrame({'A':1.,'B':pd.Timestamp('20130102'),'C':pd.Series(1,index=list(range(4)),dtype='float32'),'D':np.array([3]*4,dtype='int32'),'E':pd.Categorical(["test","train","test","train"]),'F':'foo'})
df2.dtypes
##pd.Series(range(4))
##pd.Series(np.arange(4))

df.head(2)
df.tail(3)
df.index
df.columns 
df.values
df.describe()
##基本统计量（均值 标准差 最大最小值 中位数）
df.T

df
df.sort_index(axis=1)##列 默认正序 False为反向
df.sort_index(axis=1,ascending=False)
df.sort_index(axis=0)##行
df.sort_index(axis=0,ascending=False)

df.sort_values(by='B')
df.sort_values(by='B',ascending=False)
df.sort_values(by=['A','B'],ascending=False)
##A存在两值相同时按B排列

df['A']
df[['A','C']]##提取列
df[1:4]
df.T[0:3]
df.T[0:3].T
df[-2:]
##如何提取列？用iloc/loc

 df2=pd.DataFrame(np.random.randn(5,4),columns=list('ABCD'),index=pd.date_range('20130101',periods=5))
df2 
df2[1:3]
df2.loc[2:3]##error
df2.iloc[2:4]##true
df2.iloc[2:4,1:3]##连续取行列
df2.iloc[2:4,1:5:2]
df2.loc['20130102':'20130104']##间隔取行/列
df2.loc[:,['A','C']]
df2[['A','C']]

df2.iloc[2]##默认为行
df2.iloc[1,3]##(1,3)点！！！
df.iloc[[0,2,4]]
df.iloc[[0,2,4],[1,3]]##间隔取行列
df.iloc[[0,2,4],:]

df
df2
df[df.A>0]
df2[df2.A>0]
df[df>0]
df2.loc[df2['A']>0,'A':'C']
df2.iloc[list(df2['A']>0),0:3]
##比较loc和iloc区别 名称/数字


s=pd.Series(np.arange(5),index=np.arange(5)[::-1],dtype='int64')
s[s.isin([2,4,6])]
s.isin([2,4,6])##找数值对应


df=pd.DataFrame({'vals':[1,2,3,4],'ids':['a','b','f','n'],'ids2':['a','n','c','n']});df
values=['a','b',1,3]
df.isin(values)
df[df.isin(values)]

values={'ids':['a','b'],'vals':[1,3]}
df.isin(values)
df[df.isin(values)]

values={'ids':['a','b'],'ids2':['a','c'],'vals':[1,3]}
h=df.isin(values).all(1)##1表示列 0为行
df.isin(values).all()##1表示列 0为行、
df[h]

s[s>0]
s.where(s>0)
df
df.where(df<0,-df)##全部取负数
df.where(df>0,-df)##全部取正数

df2=pd.DataFrame({'a':['one','one','two','two','two','three','four'],'b':['x','y','x','y','x','x','x'],'c':np.random.randn(7)})
np.random.randn(3)
np.random.randn(3,4).std()
np.random.randn(3,4,2)

df2.duplicated('a')
df2.duplicated('a',keep='last')##默认为False
df2.duplicated('a',keep=False)##重复为True，单独为False
df2.drop_duplicates('a')##去掉重复
df2.drop_duplicates('a',keep='last')
df2.drop_duplicates('a',keep=False)
df2.duplicated(['a','b'])
df2.drop_duplicates(['a','b'])#去掉true

df
df.mean()##默认行均值
df.mean(1)
np.mean(np.array(df))##所有值求均值

df.apply(np.cumsum)##连加
##按列连加??
df.apply(np.cumsum(df,axis=1))##连加

df.apply(lambda x:x.max()-x.min())

s=pd.Series(['A','B','C','Aaba','Baca',np.nan,'CABA','dog','cat']);s
s.str.lower()##全变小写
s.str.upper()##全变大写

df=pd.DataFrame(np.random.randn(10,4))
pieces=[df[:3],df[3:7],df[7:]]
df[1:3]
pd.concat(pieces)

df=pd.DataFrame(np.random.randn(8,4),columns=['A','B','C','D'])
s=df.iloc[3]
df.append(s)##添加至尾部
df.append(s,ignore_index=True)##true表示重新排序 默认则原始名称

df=pd.DataFrame({'A':['foo','bar','foo', 'bar','foo','bar','foo','foo'],'B':['one', 'one', 'two', 'three','two','two','one','three'],'C':np.random.randn(8),'D':np.random.randn(8)});df
df.groupby('A').sum()
df.groupby(['A','B']).mean()
df.groupby(['A','B']).agg([np.sum,np.mean,np.std])
df.groupby(['A','B']).agg({'C':np.sum,'D':lambda x:np.std(x,ddof=1)})


np.random.seed(1234)
df=pd.DataFrame(np.random.rand(50,2))
df['g']=np.random.choice(['A','B'],size=50)
df.loc[df['g']=='B',1]+=3
##第2列=“B”的+3
df.groupby('g').boxplot()##箱线图

###看不懂
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


df=pd.DataFrame({"A":["foo","foo","foo","foo","foo","bar","bar","bar","bar"],"B":["one","one","one","two", "two","one","one","two","two"],"C":["small","large","large","small","small","large","small","small","large"],"D":[1, 2,2,3,3,4,5,6,7],"E":[2,4,5,5,6,6,8,9,9]});df
table=pd.pivot_table(df,values='D',index=['A','B'],columns=['C'],aggfunc=np.sum);table
table=pd.pivot_table(df,values=['D','E'],index=['A','C'],aggfunc={'D':np.var,'E':np.mean});table
table=pd.pivot_table(df,values=['D','E'],index=['A','C'],aggfunc={'D':np.mean,'E':[min,max,np.mean]});table

foo, bar, dull, shiny, one, two = 'foo', 'bar', 'dull', 'shiny', 'one', 'two'
a = np.array([foo, foo, bar, bar, foo, foo], dtype=object)
b = np.array([one, one, two, one, two, one], dtype=object)
c = np.array([dull, dull, shiny, dull, dull, shiny], dtype=object)
pd.DataFrame({"a":a,"b":b,"c":c})
pd.crosstab(a, [b, c], rownames=['a'], colnames=['b', 'c'])##行a 列bc
df = pd.DataFrame({'A': [1, 2, 2, 2, 2], 'B': [3, 3, 4, 4, 4],'C': [1, 1, np.nan, 1, 1]})
pd.crosstab(df.A,df.B)
pd.crosstab(df['A'],df['B'],normalize='all')##全部(或者True)比例
pd.crosstab(df['A'],df['B'],normalize='columns')##行
pd.crosstab(df['A'],df['B'],normalize='index')##列
pd.crosstab(df.A,df.B,values=df.C,aggfunc=np.sum,normalize=False,margins=True)##默认为False

ages=np.array([10,15,13,12,23,25,28,59,60])
pd.cut(ages,bins=3)
pd.cut(ages,bins=[0,18,35,70])
##分组

df=pd.DataFrame({'key':list('bbacab'),'data1':range(6)});df
pd.get_dummies(df['key'])
dummies=pd.get_dummies(df['key'],prefix='key')

df=pd.DataFrame({'A':['a','b','a'],'B':['c','c','b'],'C':[1,2,3]});df
pd.get_dummies(df)##加key出错
pd.get_dummies(df,columns=['A'])

s = pd.Series(["a","b","c","a"], dtype="category");s
df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']});df
df["grade"]=df['raw_grade'].astype("category")
df["grade"].cat.categories = ["very good", "good", "very bad"]
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
 df
df.sort_values(by="grade")
df.groupby("grade").size()


ts=pd.Series(np.random.randn(1000),index=pd.date_range('1/1/2000',periods=1000));ts
ts=ts.cumsum()##按行累加
ts.plot()
df3=pd.DataFrame(np.random.rand(1000,2),columns=['B','C']).cumsum()
df3['A']=pd.Series(list(range(len(df))))

df3.plot(x='A',y='B') 

