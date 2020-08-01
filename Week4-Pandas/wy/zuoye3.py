# 作业内容 #

  #使用在校生数据集，利用python代码回答：
  import pandas as pd
  import numpy as np
  student=pd.read_excel('在校生.xls')

  ##1. 该校男生，女生分布有多少人
  student['性别'].value_counts()

  ##2. 该校各个省分别有多少人
  student['生源所在地'].value_counts()

  ##3. 该校各个省各个民族分别有多少人，分别所占的百分比是多少，并给出行列合计，结果写为csv
  stu_1=pd.crosstab(student.生源所在地, student.民族, margins=True, normalize='columns')
  stu_1.to_csv('stu_1.csv',encoding='gbk')

  ##4. 该校各个专业，各个省，男女分别有多少人，结果写为csv
  stu_2=student.pivot_table(index=['生源所在地','专业'],values=['学号'],columns=['性别'],aggfunc=len)
  stu_2.to_csv('stu_2.csv',encoding='gbk')

  ##5. 各个专业的三门课平均成绩是多少，方差是多少
  student.groupby(['专业'])[['分数1','分数2','分数3']].agg([np.mean,np.std])
  
  ##6. 所有学生的平均分和方差是多少
  student[['分数1','分数2','分数3']].values.mean()
  student[['分数1','分数2','分数3']].values.var()

  ##7. 各个专业所有课的整体平均分和方差是多少
  mean=lambda x: np.mean(x.values)
  var=lambda y: np.var(y.values)
  stu_mean=student.groupby('专业')['分数1','分数2','分数3'].apply(mean)
  
  stu_var=student.groupby('专业')['分数1','分数2','分数3'].apply(var)
  stu=pd.concat([stu_mean,stu_var],axis=1)

  ##8. 每个学生的平均分和方差是多少
  np.mean(student[['分数1','分数2','分数3']],axis=1)
  np.var(student[['分数1','分数2','分数3']],axis=1)

  ##9. 各个专业男女生的三门课平均分在80分以上的有多少人
  student['mean']=np.mean(student[['分数1','分数2','分数3']],axis=1)
  student[student['mean']>80].groupby(['专业','性别']).size()

  ##10. 找出 '学生成绩单' sheet中学生的所有其他信息，并合并到其中，把结果写为csv(使用pd的 merge 或 join 方法)
  student1=pd.read_excel('在校生.xls',1)
  student1=pd.merge(student,student1,on='学号')
  student1.to_csv('student1.csv')

  ##11. 计算合并后数据集中6门课成绩的平均分，中位数，方差，标准差，四分位点，百分位点，峰度和偏度
  del student1['mean']
  student1.iloc[:,-6:].agg([np.mean,np.std,np.var,np.median],axis=1)
  student1.iloc[:,-6:].quantile([0.25,0.75])
  student1.iloc[:,-6:].quantile(np.arange(0.01,1,0.01))
  student1.iloc[:,-6:].kurt()
  student1.iloc[:,-6:].skew()