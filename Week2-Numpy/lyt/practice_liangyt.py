### 0.3_beamer.pdf 相应代码 （至p51）

#对象操作
who
x=10.12
who
del x
who

#数值型
n=10
n
print("n=",n)

x=10.234
print(x)
print("x=%10.5f"%x)

#逻辑型
a=True;a
b=False;b
10>3
10<3
print(3)

#字符型
s='IlovePython';s
s[7]  #第一项对应0
s[2:6] #6:y不取
s+s
s*2

float('nan') #缺失值

#list
list1=[];list1
list1=['Python',786,2.23,'R',70.2]
list1
list1[0]
list1[1:3]
list1[2:]
list1*2
list1+list1[2:4]

x=[1,3,6,4,9];x
sex=['女','男','男','女','男']
sex
weight=[67,66,83,68,70]
weight

#dictionary
{}
dict1={'name':'john','code':6734,'dept':'sales'};dict1
dict1['code']
dict1.keys()
dict1.values()
dict2={'sex':sex,'weight':weight};dict2

#一维数组（向量）
import numpy as np 
np.array([1,2,3,4,5])
np.array([1,2,3,np.nan,5])
np.array(x)

np.arange(9) #元素为0~8整数
np.arange(1,9,0.5)
np.linspace(1,9,5) #两边均能取，等间隔
np.random.randint(1,9) #随机整数
np.random.rand(10) #0~1之间的随机数
np.random.randn(10) #来自标准正太分布的随机数

#二维数组（矩阵）
np.array([[1,2],[3,4],[5,6]]) #注意加中括号
A=np.arange(9).reshape((3,3));A

#数组的操作
A.shape
np.empty([3,3]) #???空数组的意义是什么
np.empty((3,3)) #???内部中括号小括号无影响
np.zeros((3,3))
np.ones((3,3))
np.eye(3)  #单位阵

### PANDAS
import pandas as pd
#创建序列
pd.Series()

X=[1,3,6,4,9]
S1=pd.Series(X);S1
S2=pd.Series(weight);S2
S3=pd.Series(sex);S3
#系列合并
pd.concat([S2,S3],axis=0) #0：承接
pd.concat([S2,S3],axis=1)
pd.concat([S1,S2,S3],axis=1) #1：并列
#系列切片
S1[2]
S3[1:4]

#生成数据框
pd.DataFrame()
#根据列表创建数据框
pd.DataFrame(X)
pd.DataFrame(X,columns=['X'],index=range(5))
pd.DataFrame(weight,columns=['weight'],index=['A','B','C','D','E'])

#根据字典创建数据框
df1=pd.DataFrame({'S1':S1,'S2':S2,'S3':S3});df1
df2=pd.DataFrame({'sex':sex,'weight':weight},index=X);df2

#增加数据框列
df2['weight2']=df2.weight**2; df2
#删除数据框列
del df2['weight2']; df2

#缺失值处理
df3=pd.DataFrame({'S2':S2,'S3':S3},index=S1);df3 #缺失值产生原因：index不能完全与数据下标匹配
df3.isnull()
df3.isnull().sum()
df3.dropna()
#df3.dropna(how = 'all')

#数据框排序
df3.sort_index() #自小到大
df3.sort_values(by='S3')

###???数据
#读取 csv 文件的方法
BSdata=pd.read_csv(”../data/BSdata.csv”,encoding='utf-8')
BSdata[6:9]

#读取 Excel 格式数据
BSdata=pd.read_excel('../data/DaPy_data.xlsx','BSdata');BSdata[-5:]

#pandas 数据集的保存
BSdata.to_csv('BSdata1.csv')

#显示基本信息
BSdata.info()  #显示数据结构
BSdata.head()  #显示数据框前5行
BSdata.tail()  #显示数据框后5行

#数据框列名（变量名）
BSdata.columns
#数据框行名（样品名）
BSdata.index
#数据框维度
BSdata.shape
BSdata.shape[0] # 行数
BSdata.shape[1] # 列数
#数据框值（数组）
BSdata.values

BSdata.身高 #取一列数据，BSdata['身高']
BSdata[['身高','体重']]
BSdata.iloc[:,2]    
BSdata.iloc[:,2:4]  #第3、4列
# dat.iloc［i,］表示dat的第i行数据向量，而dat.iloc[,j]表示dat的第j列数据向量

#选取样本与变量
BSdata.loc[3]
BSdata.loc[3:5]
BSdata.loc[:3,['身高','体重']]
BSdata.iloc[:3,:5]

#条件选取
BSdata[BSdata['身高']>180]
BSdata[(BSdata['身高']>180)&(BSdata['体重']<80)]  # &:且

#生成新的数据框
BSdata['体重指数']=BSdata['体重']/(BSdata['身高']/100)**2
round(BSdata[:5],2) 

#数据框转置.T
BSdata.iloc[:3,:5].T

#数据框的运算
pd.concat([BSdata.身高, BSdata.体重],axis=0)  # 按行合并，axis=0
pd.concat([BSdata.身高, BSdata.体重],axis=1)  # 按列合并，axis=1


