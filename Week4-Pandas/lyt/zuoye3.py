########## 作业内容 ###########
## 使用在校生数据集，利用python代码回答：

import numpy as np 
import pandas as pd 
students=pd.read_excel('在校生.xls')
students.head()

  ###1. 该校男生，女生分布有多少人

students.性别.value_counts()

  ###2. 该校各个省分别有多少人

students.生源所在地.value_counts()

  ###3. 该校各个省各个民族分别有多少人，分别所占的百分比是多少，并给出行列合计，结果写为csv

counts = pd.crosstab(students.生源所在地,students.民族,margins=True,margins_name='合计')
counts.to_csv('03_1.csv',encoding='gbk')

percentage = pd.crosstab(students.生源所在地,students.民族,normalize='all',margins=True,margins_name='合计')
percentage.to_csv('03_2.csv',encoding='gbk')

  ###4. 该校各个专业，各个省，男女分别有多少人，结果写为csv

sex_counts = pd.crosstab([students.专业,students.生源所在地],students.性别)
sex_counts.to_csv('04.csv',encoding='gbk')

#???students.pivot_table(index=['专业','生源所在地'],values=['性别'],aggfunc=np.sum) 
#####???怎么用透视图，当values时定性数据时

  ###5. 各个专业的三门课平均成绩是多少，方差是多少

students.groupby('专业')['分数1','分数2','分数3'].agg(['mean','var'])   

  ###6. 所有学生的平均分和方差是多少

score = students.iloc[:,-3:]
score_mean = score.values.mean()
score_var = score.values.var()
print('平均分:',score_mean,'方差:',score_var)

  ###7. 各个专业所有课的整体平均分和方差是多少

major_mean = students.groupby('专业')['分数1','分数2','分数3'].agg('mean').mean(axis=1)
var_new = lambda x: np.var(x.values)
major_var = students.groupby('专业')['分数1','分数2','分数3'].apply(var_new)       ##必须用apply():整体传输；agg()结果是多列
pd.concat([major_mean,major_var],axis=1,keys=['整体平均分','整体方差'])

  ###8. 每个学生的平均分和方差是多少

student_mean = students[['分数1','分数2','分数3']].mean(axis=1)
student_var = students.groupby('学号')['分数1','分数2','分数3'].apply(var_new)
student_var = student_var.reset_index(drop=True)
pd.concat([student_mean,student_var],axis=1,keys=['平均分','方差'])

  ###9. 各个专业男女生的三门课平均分在80分以上的有多少人

students['平均分'] = students[['分数1','分数2','分数3']].mean(axis=1)
s1 = students.loc[students['平均分']>80]
s1.groupby(['专业','性别']).size().to_frame()

  ###10. 找出 '学生成绩单' sheet中学生的所有其他信息，并合并到其中，把结果写为csv(使用pd的 merge 或 join 方法)

report_card = pd.read_excel('在校生.xls',sheet_name='学生成绩单')
report_card.head()
students = students.drop(['平均分'],axis=1)
pd.merge(students,report_card).to_csv('10.csv',encoding='gbk')

  ###11. 计算合并后数据集中6门课成绩的平均分，中位数，方差，标准差，四分位点，百分位点，峰度和偏度

new_data = pd.read_csv('10.csv',encoding='gbk').iloc[:,-6:]
new_data.head() 
new_data.describe().iloc[[1,2,4,5,6],:]   #平均分、标准差、四分位点、中位数
var = new_data.var() ;var               #方差
percent = new_data.quantile(np.arange(0.01,1,0.01)) ;percent    #百分位点
skew = new_data.skew() ;skew            #偏度
kurt = new_data.kurt() ;kurt            #峰度
