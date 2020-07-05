## 作业内容 ##

# 1. 分别创建一个列表,元组，字典，并提取其子集
a=[1,2,3,4]  #列表
a[1:3]

b=(1,2,3,4,5,6) #元组
b[2:4]

c={'q':80, 'w':70, 'e':90} #字典
c['w']

# 2. 利用 numpy 生成一个一维数组和二维数组
import numpy as np
a=np.array([1,2,3,4,5]) #一维数组
b=np.array([[1,2],[3,4],[5,6]]) #二维数组

# 3. 0-100之间生成等步长的201个数；把列表 [1,4,9] 重复10遍
import numpy as np
np.linspace(0,100,201)
np.linspace([1,4,9,],[1,4,9],10)

# 4. 随机生成8乘以20的整数array，并得到其转置数组，对其转置数组提取前10行，第2,5,8列数据
import numpy as np
a=np.random.randint(1,100,size=(8,20));a
b=a.T
b[:10,[1,4,7]]

# 5. 利用pandas把data文件夹中的 数据.xls 导入，要求使用相对路径
import pandas as pd
d1=pd.read_excel('../data/数据.xls',0)
d1

# 6. 显示 数据.xls 数据集的结构，前5条数据，后5条数据，所有变量名，及其维度
d1.info()
d1.head()
d1.tail()
d1.columns
d1.shape

# 7. 对数据集增加一个变量HRsq，其为 HR数据的平方
d1['HRsq']=d1['HR']**2
d1

# 8. 对 数据.xls 数据集 进行子集的提取：
  ## 1. 删除有缺失值的年份
  d2=d1.dropna()
  ## 2. 提取逢5，逢0年份的Year, GDP, KR, HRsq，CPI 变量的数据
  d3=d2[d2['Year']%5==0][['Year','GDP','KR','HRsq','CPI']];d3
  ## 3. 提取逢2，逢8年份的Year, GDP, KR, HRsq，CPI 变量的数据
  data1=d2[d2['Year']%10==2][['Year','GDP','KR','HRsq','CPI']];data1
  data2=d2[d2['Year']%10==8][['Year','GDP','KR','HRsq','CPI']];data2
  d4=pd.concat([data1,data2],axis=0);d4
  ## 4. 对2和3得到的数据集按行进行合并，并按年份进行排序  1111 
  d5=pd.concat([d3,d4],axis=0);d5
  d6=d5.sort_values(by='Year');d6
  ## 5. 提取1978年之后的数据
  d7=d5[d5['Year']>1978];d7
  ## 6. 提取1978年之后且 KR 变量在 1~1.2之间的数据
  d8=d5[(d5['Year']>1978)&(d5['KR']>1)&(d5['KR']<1.2)];d8
# 9. 保存数据为csv和excel
  ## 1. 写出第8题中第4问得到的子集到data文件夹
  d6.to_csv('../data/data6.csv')
  d6.to_excel('../data/data6.xlsx')
  ## 2. 写出第8题中第6问得到的子集到data文件夹
  d8.to_csv('../data/data8.csv')
  d8.to_excel('../data/data8.xlsx')