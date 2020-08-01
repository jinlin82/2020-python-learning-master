###作业内容
###使用在校生数据集，利用python代码回答：
import pandas as pd 
import numpy as np 
df=pd.read_excel('在校生.xls');df
df.columns
###1. 该校男生，女生分布有多少人
df.性别.value_counts()

##2. 该校各个省分别有多少人
df.生源所在地.value_counts()

  # 3. 该校各个省各个民族分别有多少人，分别所占的百分比是多少，并给出行列合计，结果写为csv
df.groupby(['生源所在地','民族']).size()
df.groupby(['生源所在地','民族']).size()/df.shape[0]

df1=pd.crosstab(df['生源所在地'],df['民族'],margins=True,margins_name='合计').round(3);df1##人数
df1=pd.crosstab(df['生源所在地'],df['民族'],margins=True,margins_name='合计',normalize='all').round(3);df1##比例
df1.to_csv('./hw3/new1.csv')

# 4. 该校各个专业，各个省，男女分别有多少人，结果写为csv
df2=pd.DataFrame(df.groupby(['专业','生源所在地'])['性别'].value_counts());df2
df2.to_csv('./hw3/new2.csv')

# pd.crosstab([df['专业'],df['生源所在地']],df['性别'])


  # 5. 各个专业的三门课平均成绩是多少，方差是多少

# df.groupby(['专业'])[['分数1','分数2','分数3']].agg([np.mean,np.var])
df.pivot_table(index=['专业'],values=['分数1','分数2','分数3'],aggfunc=[np.mean,np.var])




  # 6. 所有学生的平均分和方差是多少

df[['分数1','分数2','分数3']].values.mean()
df[['分数1','分数2','分数3']].values.var()

# df[['总平均分']].agg([np.mean,np.var])
# 这里为均值的方差

h1=list(df['分数1'])
h2=list(df['分数2'])
h3=list(df['分数3'])
h=h1+h2+h3
np.mean(h)
np.var(h)


  # 7. 各个专业所有课的整体平均分和方差是多少
# df[['分数1','分数2','分数3']].agg([np.mean,np.var])
# df[['分数1','分数2','分数3']].mean()
# df[['分数1','分数2','分数3']].var()

# df['总平均分']=df[['分数1','分数2','分数3']].apply([np.mean],axis=1)
# df.groupby(['专业'])['总平均分'].agg([np.mean,np.var])
# 总平均分的均值和方差

d1=lambda x:np.mean(x.values)
d2=lambda x:np.var(x.values)
d3=df.groupby(['专业'])['分数1','分数2','分数3'].apply(d1)
d4=df.groupby(['专业'])['分数1','分数2','分数3'].apply(d2)
pd.DataFrame({'mean':d3,'var':d4})

  # 8. 每个学生的平均分和方差是多少
df3=df[['分数1','分数2','分数3']].apply([np.mean,np.var], axis=1)
df3
# df3=df[['分数1','分数2','分数3']].agg([np.mean,np.var], axis=1)
# apply和agg均可以


  # 9. 各个专业男女生的三门课平均分在80分以上的有多少人
df['总平均分']=df[['分数1','分数2','分数3']].apply([np.mean],axis=1)
df[df['总平均分']>80].groupby(['专业','性别']).size()

  # 10. 找出 '学生成绩单' sheet中学生的所有其他信息，并合并到其中，把结果写为csv(使用pd的 merge 或 join 方法)
df
df4=pd.read_excel('在校生.xls',1);df4
df5=pd.merge(df,df4,how='outer');df5
df5.to_csv('./hw3/new3.csv')

  # 11. 计算合并后数据集中6门课成绩的平均分，中位数，方差，标准差，四分位点，百分位点，峰度和偏度

# df5[['分数1','分数2','分数3','分数4','分数5','分数6']].describe().T
df5[['分数1','分数2','分数3','分数4','分数5','分数6']].agg([np.mean,np.median,np.var,np.std])
df5[['分数1','分数2','分数3','分数4','分数5','分数6']].quantile([0.25,0.5,0.75])
df5[['分数1','分数2','分数3','分数4','分数5','分数6']].quantile(np.arange(0.01,1,0.01))
df5[['分数1','分数2','分数3','分数4','分数5','分数6']].kurt()
df5[['分数1','分数2','分数3','分数4','分数5','分数6']].skew()





