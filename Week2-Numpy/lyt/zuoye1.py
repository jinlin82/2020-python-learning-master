## 作业内容 ##

## 1. 分别创建一个列表,元组，字典，并提取其子集 

list=[1,2,3,4]   #列表
list[1:3]

tuple=(1,2,3,4)    #元组
tuple[1:3]

dict={'A':11,'B':22,'C':33}   #字典
dict['A']


##  2. 利用 numpy 生成一个一维数组和二维数组

import numpy as np 
x=np.array([1,2,3,4]);x              #一维数组
y=np.array([[1,2],[3,4],[5,6]]);y    #二维数组


##  3. 0-100之间生成等步长的201个数；把列表 [1,4,9] 重复10遍

import numpy as np 
np.linspace(0,100,num=201)

a=[1,4,9]
A=a*10;A


##  4. 随机生成8乘以20的整数array，并得到其转置数组，对其转置数组提取前10行，第2,5,8列数据

import numpy as np 
r=np.random.randint(1,100,size=160)
R=r.reshape((8,20),order='c');R

RT=R.T;RT     #转置

import pandas as pd
rt1=RT[:10,1]  #第2列数据
rt2=RT[:10,4]  #第5列数据
rt3=RT[:10,7]  #第8列数据
rt_1=pd.Series(rt1)
rt_2=pd.Series(rt2)
rt_3=pd.Series(rt3)
rtc=pd.concat([rt_1,rt_2,rt_3],axis=1);rtc


##  5. 利用pandas把data文件夹中的 数据.xls 导入，要求使用相对路径





##  6. 显示 数据.xls 数据集的结构，前5条数据，后5条数据，所有变量名，及其维度

##  7. 对数据集增加一个变量HRsq，其为 HR数据的平方

##  8. 对 数据.xls 数据集 进行子集的提取：

###     1. 删除有缺失值的年份

###     2. 提取逢5，逢0年份的Year, GDP, KR, HRsq，CPI 变量的数据

###     3. 提取逢2，逢8年份的Year, GDP, KR, HRsq，CPI 变量的数据

###     4. 对2和3得到的数据集按行进行合并，并按年份进行排序  1111 

###     5. 提取1978年之后的数据

###     6. 提取1978年之后且 KR 变量在 1~1.2之间的数据

##  9. 保存数据为csv和excel

###     1. 写出第8题中第4问得到的子集到data文件夹

###     2. 写出第8题中第6问得到的子集到data文件夹
