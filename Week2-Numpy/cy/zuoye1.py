#1. 分别创建一个列表,元组，字典，并提取其子集

list1 = [1,2,'hello'] #列表
list1[1:3]

tuple = (1,2,'hello') #元组
tuple[1:3]

sex = ['男','男','女'] #字典
height = [183,181,165]
dict1 = {'sex':sex,'height':height};dict1

#2. 利用 numpy 生成一个一维数组和二维数组

import numpy as np 
np.array([1,2,3,np.nan,5]) #一维数组
np.array([[1,2],[3,4],[5,6]]) #二维数组

#3. 0-100之间生成等步长的201个数；把列表 [1,4,9] 重复10遍

import numpy as np 
np.linspace(0,100,201)
list2 = [1,4,9]
list2*10

#4. 随机生成8乘以20的整数array，并得到其转置数组，对其转置数组提取前10行，第2,5,8列数据

import numpy as np 
import pandas as pd
A = np.arange(160).reshape((8,20))
AT = A.T 
#AT.shape
AT2 = AT[:10,1]
S2 = pd.Series(AT2)
AT5 = AT[:10,4]
S5 = pd.Series(AT5)
AT8 = AT[:10,7]
S8 = pd.Series(AT8)
S = pd.concat([S2,S5,S8],axis = 1);S

#5. 利用pandas把data文件夹中的 数据.xls 导入，要求使用相对路径

import pandas as pd 

workdata = pd.read_excel('../data/数据.xls');

#6. 显示 数据.xls 数据集的结构，前5条数据，后5条数据，所有变量名，及其维度

workdata.info()
workdata.head()
workdata.tail()
workdata.index #变量名
workdata.shape #行列数
workdata.shape[0] #行
workdata.shape[1] #列

#7. 对数据集增加一个变量HRsq，其为 HR数据的平方

workdata['HRsq'] = workdata.HR**2

#8. 对 数据.xls 数据集 进行子集的提取：

##1. 删除有缺失值的年份

workdata.isnull()
workdata.isnull().sum()
workdata1 = workdata.dropna()
workdata1.isnull().sum()

##2. 提取逢5，逢0年份的Year, GDP, KR, HRsq，CPI 变量的数据


###import sys # 保存当前的sys.stdout状态, 开始捕获当前的输出 
###current = sys.stdout 
###f = open('./cy.csv', 'w') # 这一步实际是sys.stdout.write, 当sys捕获到了print
#输出的时候, 就写入f里面 
#####
#####
###sys.stdout = f # 恢复状态, 之后的print内容都不捕获了 
###sys.stdout = current


###这个运行不了？？？
#workdata1.loc[[i],['Year','GDP','KR','HRsq','CPI']]
#workdata1.loc[2,['Year','GDP','KR','HRsq','CPI']]

SS1 = pd.DataFrame()
for i in range(len(workdata1)):
    S = str(workdata1.Year.iloc[i])[3] 
    if(S =='5') |(S == '0'):
        SS1 = pd.concat([SS1, workdata1.iloc[[i],[0,1,8,9,10]]],axis=0)
print(SS1)
SS1

##3. 提取逢2，逢8年份的Year, GDP, KR, HRsq，CPI 变量的数据

SS2 = pd.DataFrame()
for i in range(len(workdata1)):
    S = str(workdata1.Year.iloc[i])[3] 
    if(S =='2') |(S == '8'):
        SS2 = pd.concat([SS2, workdata1.iloc[[i],[0,1,8,9,10]]],axis=0)
print(SS2)
SS2

##4. 对2和3得到的数据集按行进行合并，并按年份进行排序  

dataT4 = pd.concat([SS1,SS2],axis=0)
dataT4 = dataT4.sort_values(by='Year')
dataT4

##5. 提取1978年之后的数据

data1 = workdata1[workdata1.Year>1978]

##6. 提取1978年之后且 KR 变量在 1~1.2之间的数据

data2 = workdata1[(workdata1.Year>1978)&(workdata1.KR>=1)&(workdata1.KR<=1.2)]

#9. 保存数据为csv和excel

##1. 写出第8题中第4问得到的子集到data文件夹

dataT4.to_csv('data2T9_1.csv')
dataT4.to_excel('data2T9_1.xlsx')

##2. 写出第8题中第6问得到的子集到data文件夹

data2.to_csv('data2T9_2.csv')
data2.to_excel('data2T9_2.xlsx')

