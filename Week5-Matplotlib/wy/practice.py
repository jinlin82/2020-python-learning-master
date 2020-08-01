# 05.pdf #

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

#特殊统计图的绘制

##初等函数图
x=np.linspace(0,2*math.pi);x
#fig,ax=plt.subplots(2,2,figsize=(15,12))

plt.plot(x,np.sin(x))
plt.plot(x,np.cos(x))
plt.plot(x,np.log(x))
plt.plot(x,np.exp(x))

##极坐标图（加公式）
t=np.linspace(0,2*math.pi);t
x=2*np.sin(t)
y=3*np.cos(t)
plt.plot(x,y)
plt.text(0,0,r'$\frac{x^2}{2}+\frac{y^2}{3}=1$',fontsize=15)

##气泡图
BSdata=pd.read_csv('BSdata.csv')
plt.scatter(BSdata['身高'],BSdata['体重'],s=BSdata['支出'])

##三维曲面图
fig=plt.figure()
ax=Axes3D(fig)
x=np.linspace(-4,4,20)
y=np.linspace(-4,4,20)
x,y=np.meshgrid(x,y)
z=np.sqrt(x**2+y**2)
ax.plot_surface(x,y,z)

##三维散点图
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(BSdata.身高, BSdata.体重, BSdata.支出,s=50*np.random.rand(52)) #三维气泡图
ax.scatter(BSdata.身高, BSdata.体重, BSdata.支出)


#seaborn统计绘图

##箱线图
plt.rcParams['font.sans-serif']=['KaiTi']
sns.boxplot(x=BSdata.身高) #绘制箱线图
sns.boxplot(y=BSdata.身高) #竖着放的箱线图，也就是将x换为y
sns.boxplot(x='性别',y='身高',data=BSdata) #分组绘制箱线图，分组因子是性别，在x轴不同位置绘制
sns.boxplot(x='开设',y='支出',hue='性别',data=BSdata) #根据开设和性别两个分类看支出的情况
plt.text(80,1,r'$\bar x$')
sns.boxplot(y='开设',x='支出',hue='性别',data=BSdata)

##小提琴图
sns.violinplot(x='性别',y='身高',data=BSdata)
sns.violinplot(x='开设',y='支出',hue='性别',data=BSdata)

##点图
sns.stripplot(x='性别',y='身高',data=BSdata)
sns.stripplot(x='性别',y='身高',data=BSdata,jitter=True)
sns.stripplot(y='性别',x='身高',data=BSdata,jitter=True)

##条图
sns.barplot(x='性别',y='身高',data=BSdata,ci=0,palette='Blues_d')

##计数图
sns.countplot(x='性别',data=BSdata)
sns.countplot(y='开设',data=BSdata)
sns.countplot(x='性别',hue='开设',data=BSdata)

##分组关系图
sns.catplot(x='性别',col='开设',col_wrap=3,data=BSdata,kind='count',height=2.5,aspect=.8)

##概率分布图
sns.distplot(BSdata.身高,kde=True,bins=20,rug=True)
sns.jointplot(x='身高',y='体重',data=BSdata)
sns.pairplot(BSdata[['身高','体重','支出']])

#ggplot绘图系统

##qplot快速绘图
from ggplot import *
plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus']=False

###直方图
qplot('身高',data=BSdata,geom='histogram')

###条形图
qplot('开设',data=BSdata,geom='bar')

###散点图
qplot('身高','体重',data=BSdata,color='性别')


##ggplot基本绘图
GP=ggplot(aes(x='身高',y='体重'),data=BSdata);GP #绘制坐标轴

ggplot(BSdata,aes(x='身高'))+geom_histogram() #直方图

GP+geom_point() #散点图

ggplot(BSdata,aes(x='支出'))+geom_line(aes(y='身高')) #线图

ggplot(BSdata,aes(x='身高',y='体重'))+geom_point()+facet_wrap('性别') #分面图
ggplot(BSdata,aes(x='身高',y='体重'))+geom_point()+facet_wrap('性别',nrow=1,ncol=2)

ggplot(BSdata,aes(x='身高',y='体重',color='性别'))+geom_point()+theme_bw() #图形主题



