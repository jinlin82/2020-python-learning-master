# python数据可视化分析
## 特殊统计图的绘制
### 初等函数图
import math
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus']=False
x=np.linspace(0,2*math.pi);x

plt.plot(x,np.sin(x))
plt.plot(x,np.cos(x))
plt.plot(x,np.log(x))
plt.plot(x,np.exp(x))

### 极坐标图
t=np.linspace(0,2*math.pi)
x=2*np.sin(t)
y=3*np.cos(t)
plt.plot(x,y)
plt.text(0,0,r'$\frac{x^2}{2}+\frac{y^2}{3}=1$',fontsize=15) # 在图中加文字注释

### 气泡图
import pandas as pd
BSdata=pd.read_csv('C:/github_repository/2020-python-learning-master/Week4-Pandas/lmk/data/BSdata.csv',encoding='utf-8')
plt.scatter(BSdata['身高'],BSdata['体重'],s=BSdata['支出']) # s为气泡大小

### 三维曲面图
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure() # 画布
ax=Axes3D(fig) # 坐标系
X=np.linspace(-4,4,20)
Y=np.linspace(-4,4,20)
X,Y=np.meshgrid(X,Y) # 将X,Y向量变成交织的格子
Z=np.sqrt(X**2+Y**2)
ax.plot_surface(X,Y,Z)

### 三维散点图
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(BSdata['身高'],BSdata['体重'],BSdata['支出'])
ax.scatter(BSdata['身高'],BSdata['体重'],BSdata['支出'],s=50*np.random.rand(52)) # s设置点的大小

## seaborn统计绘图
import seaborn as sns

### 箱线图
sns.boxplot(x=BSdata['身高']) # 横放
sns.boxplot(y=BSdata['身高']) # 竖放
sns.boxplot(x='性别',y='身高',data=BSdata) # 分组绘制，分组因子是性别
sns.boxplot(x='开设',y='支出',hue='性别',data=BSdata) # 分组箱线图

### 小提琴图
sns.violinplot(x='性别',y='身高',data=BSdata)
sns.violinplot(x='开设',y='支出',hue='性别',data=BSdata)

### 点图
sns.stripplot(x='性别',y='身高',data=BSdata)
sns.stripplot(x='性别',y='身高',data=BSdata,jitter=True)
sns.stripplot(y='性别',x='身高',data=BSdata,jitter=True) # jitter=True表示如果有两个点是相同的，则将它们分开一些

### 条图
sns.barplot(x='性别',y='身高',data=BSdata,ci=0,palette='Blues_d')

### 计数图
sns.countplot(x='性别',data=BSdata)
sns.countplot(y='开设',data=BSdata)
sns.countplot(x='性别',hue='开设',data=BSdata)

### 分组关系图
sns.catplot(x='性别',col='开设',col_wrap=3,data=BSdata,kind='count',height=2.5,aspect=0.8) # col_wrap控制一行有几个图

### 概率分布图
sns.distplot(BSdata['身高'],kde=True,bins=20,rug=True) # rug控制是否画样本点
sns.jointplot(x='身高',y='体重',data=BSdata) # 双变量
sns.pairplot(BSdata[['身高','体重','支出']]) # 多变量（默认对角线为直方图，非对角线为散点图）

## ggplot绘图系统
### qplot快速制图
from ggplot import *
#### 直方图
qplot('身高',data=BSdata,geom='histogram')

#### 条形图
qplot('开设',data=BSdata,geom='bar')

#### 散点图
qplot('身高','体重',data=BSdata,color='性别')

### ggplot基本绘图
GP=ggplot(aes(x='身高',y='体重'),data=BSdata);GP # 绘制直角坐标系
#### 直方图
ggplot(BSdata,aes(x='身高'))+geom_histogram()

#### 散点图
ggplot(BSdata,aes(x='身高',y='体重'))+geom_point()

#### 线图
ggplot(BSdata,aes(x='支出'))+geom_line(aes(y='身高'))

#### 分面图
ggplot(BSdata,aes(x='身高',y='体重'))+geom_point()+facet_wrap('性别')
ggplot(BSdata,aes(x='身高',y='体重'))+geom_point()+facet_wrap('性别',nrow=1,ncol=2)

#### 图形主题
ggplot(BSdata,aes(x='身高',y='体重',color='性别'))+geom_point()+theme_bw()
ggplot(BSdata,aes(x='身高',y='体重',color='性别'))+geom_point()+theme_xkcd() # 中文支持不好