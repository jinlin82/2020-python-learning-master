# 作业内容
## 使用在校生数据集，利用python代码回答：
### 1. 该校男生，女生分别有多少人

import pandas as pd
student=pd.read_excel('./data/在校生.xls',encoding='utf-8')
student.性别.value_counts()

### 2. 该校各个省分别有多少人

student.生源所在地.value_counts()

### 3. 该校各个省各个民族分别有多少人，分别所占的百分比是多少，并给出行列合计，结果写为csv

nation=pd.crosstab(student['生源所在地'],student['民族'],margins=True,margins_name='合计');nation
nation_ratio=pd.crosstab(student['生源所在地'],student['民族'],normalize=True,margins=True,margins_name='合计');nation_ratio
nation.to_csv('nation.csv',encoding='gb2312')
nation_ratio.to_csv('nation_ratio.csv',encoding='gb2312')

### 4. 该校各个专业，各个省，男女分别有多少人，结果写为csv

major=student.pivot_table(values='学号',index=['专业','生源所在地'],columns='性别',aggfunc=len);major
major.to_csv('major.csv',encoding='gb2312')

### 5. 各个专业的三门课平均成绩是多少，方差是多少

import numpy as np
student.groupby('专业')['分数1','分数2','分数3'].agg([np.mean,np.var])

### 6. 所有学生的平均分和方差是多少

np.mean(student[['分数1','分数2','分数3']].values)
np.var(student[['分数1','分数2','分数3']].values)

### 7. 各个专业所有课的整体平均分和方差是多少

mean=student.groupby('专业')['分数1','分数2','分数3'].apply(lambda x:x.values.mean())
var=student.groupby('专业')['分数1','分数2','分数3'].apply(lambda x:x.values.var(ddof=1))
# np.var()的对象是array,默认计算总体方差,n-ddof为自由度,样本方差应设置ddof=1
pd.concat([mean,var],axis=1,keys=['mean','var'])

zcx=student[student['专业']=='侦查学'].iloc[:,-3:]
# var1=zcx.apply(lambda x:x.values.var())
zcx.values.var()
zcx1=zcx.values # ndarray 对所有数求方差
zcx1.var()
zcx.var() # dataframe 对三列分别求方差

np.array([1,2,3]).var()
np.array([1,2,3]).var(ddof=1)
pd.DataFrame([1,2,3]).var()

### 8. 每个学生的平均分和方差是多少

student[['分数1','分数2','分数3']].agg([np.mean,np.var],axis=1)

### 9. 各个专业男女生的三门课平均分在80分以上的有多少人

student['score_mean']=student[['分数1','分数2','分数3']].apply(np.mean,axis=1)
pd.DataFrame(student.loc[(student['score_mean']>80)].groupby(['专业','性别']).size(),columns=['人数'])

### 10. 找出 '学生成绩单' sheet中学生的所有其他信息，并合并到其中，把结果写为csv(使用pd的 merge 或 join 方法)

del student['score_mean']
scores=pd.read_excel('./data/在校生.xls','学生成绩单',encoding='utf-8')
summation=pd.merge(student,scores,on='学号')
summation.to_csv('summation.csv',encoding='gb2312')

### 11. 计算合并后数据集中6门课成绩的平均分，中位数，方差，标准差，四分位点，百分位点，峰度和偏度

summation[['分数1','分数2','分数3','分数4','分数5','分数6']].agg([np.mean,np.median,np.var,np.std])
summation[['分数1','分数2','分数3','分数4','分数5','分数6']].quantile([0.25,0.75])
summation[['分数1','分数2','分数3','分数4','分数5','分数6']].quantile(np.arange(0.01,1.01,0.01))
summation[['分数1','分数2','分数3','分数4','分数5','分数6']].kurt()
summation[['分数1','分数2','分数3','分数4','分数5','分数6']].skew()