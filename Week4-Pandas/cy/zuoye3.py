
import numpy as np 
import pandas as pd 
students = pd.read_excel('./在校生.xls',0)
students.head()

#1. 该校男生，女生分布有多少人

sex = students.性别.value_counts();sex

#2. 该校各个省分别有多少人

province = students.生源所在地.value_counts();province

#3. 该校各个省各个民族分别有多少人，分别所占的百分比是多少，并给出行列合计，结果写为csv

#pro_peoples = students.groupby(['生源所在地','民族']).count().loc[:,['学号']]
#pro_peoples.columns = ['人数']
#proportion = round(pro_peoples/#pro_peoples.groupby(level=0).sum()*100,2)
#pro_peoples['百分比'] = proportion
#pro_peoples.to_csv('./province_peoples.csv')

pp = pd.crosstab(students['生源所在地'],students['民族'],margins=True,margins_name='合计')
pp.to_csv('./zuoye3_data/各省民族频数表.csv',encoding='gb2312')
ppp = pd.crosstab(students['生源所在地'],students['民族'],margins=True,margins_name='合计',normalize='all')
pp.to_csv('./zuoye3_data/各省民族频率表.csv',encoding='gb2312')


#4. 该校各个专业，各个省，男女分别有多少人，结果写为csv

maj_pro_sex = students.groupby(['专业','生源所在地','性别']).count().loc[:,['学号']]
maj_pro_sex.columns = ['人数']
maj_pro_sex.to_csv('./zuoye3_data/各专业各省男女人数.csv',encoding='gb2312')

#5. 各个专业的三门课平均成绩是多少，方差是多少

major_m_s = students.groupby('专业')['分数1','分数2','分数3'].agg(['mean','std'])
major_m_s

#6. 所有学生的平均分和方差是多少

scores_all = students[['分数1','分数2','分数3']]
score_all_mean = scores_all.values.mean()
score_all_var = scores_all.values.var()
print('所有学生的平均分和方差分别为：',score_all_mean,score_all_var)

#7. 各个专业所有课的整体平均分和方差是多少

all_mean = lambda x:x.values.mean()
all_var = lambda x:x.values.mean()
major_all_mean = students.groupby('专业')['分数1','分数2','分数3'].apply(all_mean)
major_all_var = students.groupby('专业')['分数1','分数2','分数3'].apply(all_var)
major_all_mv = pd.concat([major_all_mean,major_all_var],axis=1) 
major_all_mv.columns = ['平均分','方差']
major_all_mv

#8. 每个学生的平均分和方差是多少

## way 1 
score_every_mean = scores_all.mean(axis=1)
score_every_var = scores_all.var(axis=1)
students['平均分'] = score_every_mean
students['方差'] = score_every_var
students[['学号','平均分','方差']]

## way 2
students[['分数1','分数2','分数3']].agg(['mean','var'],axis=1)


#9. 各个专业男女生的三门课平均分在80分以上的有多少人

m_over_80 = students[students.平均分>80].groupby(['专业','性别'])[['学号']].count()
m_over_80.columns = ['人数']
m_over_80

#10. 找出 '学生成绩单' sheet中学生的所有其他信息，并合并到其中，把结果写为csv(使用pd的 merge 或 join 方法)

students1 = students
del students1['平均分']
del students1['方差']
students2 = pd.read_excel('./在校生.xls',1)
student_merge = pd.merge(students1,students2,on='学号')
student_merge.to_csv('./zuoye3_data/合并后在校生.csv',encoding='gb2312')

#11. 计算合并后数据集中6门课成绩的平均分，中位数，方差，标准差，四分位点，百分位点，峰度和偏度

student_merge[['分数1','分数2','分数3','分数4','分数5','分数6']].agg(['mean','median','var','std','kurt','skew'])

student_merge[['分数1','分数2','分数3','分数4','分数5','分数6']].quantile([0.25,0.5,0.75])
student_merge[['分数1','分数2','分数3','分数4','分数5','分数6']].quantile(np.arange(0.01,1,0.01))
