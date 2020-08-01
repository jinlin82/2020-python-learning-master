###### 04_beamer.pdf #######

### 常用的绘图函数
import matplotlib.pyplot as plt          #基本绘图包
plt.rcParams['font.sans-serif']=['KaiTi']     #SimHei黑体
plt.rcParams['axes.unicode_minus']=False     #正常显示图中负号
# plt.figure(figsize=(6,5))     #图形大小

### 常见的统计作图函数
X=['A','B','C','D','E','F','G']
Y=[1,4,7,3,2,5,6]
plt.bar(X,Y)    #条图
plt.savefig('abc',format='pdf')

plt.pie(Y,labels=X)   #饼图
plt.show()      #两行选中同时运行

plt.plot(X,Y)    #线图

plt.hist(BSdata.身高)      #频数直方图
plt.hist(BSdata.身高,density=True)   #频率直方图

plt.scatter(BSdata.身高,BSdata.体重)   #散点图

### 标题、标签、标尺及颜色
plt.ylim(0,8);
plt.xlabel('names');plt.ylabel('values');
plt.xticks(range(len(X)),X)

### 线型和符号
plt.plot(X,Y,linestyle='--',marker='o')

plt.plot(X,Y,'o--')
plt.axvline(x=1)
plt.axhline(y=4)

plt.plot(X,Y)
plt.text(2,7,'peakpoint')   #在（x,y）处添加用 labels 指定的文字

###  legend 函数给图形加图例
plt.plot(X,Y,label=u'折线')
plt.legend()                 #图例

s=[0.1,0.4,0.7,0.3,0.2,0.5,0.6]
plt.bar(X,Y,yerr=s,error_kw={'capsize':5})  #误差条图

### 多图

#'''一行绘制两个图形'''
plt.figure(figsize=(12,6))
plt.subplot(1,2,1); plt.bar(X,Y)
plt.subplot(1,2,2); plt.plot(Y)     #一行

 #'''一列绘制两个图形'''
plt.figure(figsize=(7,10))
plt.subplot(2,1,1); plt.bar(X,Y)
plt.subplot(2,1,2); plt.plot(Y)     #一列

 #'''一页绘制两个图形'''
fig,ax = plt.subplots(1,2,figsize=(14,6))
ax[0].bar(X,Y)
ax[1].plot(X,Y)                     #一页

 #'''一页绘制四个图形'''
fig,ax=plt.subplots(2,2,figsize=(15,10))
ax[0,0].bar(X,Y); ax[0,1].pie(Y,labels=X)
ax[1,0].plot(Y); ax[1,1].plot(Y,'.-',linewidth=3)

### 基于pandas的绘图 （kind=...）
DataFrame.plot(kind='line')

BSdata['体重'].plot(kind='line')
BSdata['体重'].plot(kind='hist')
BSdata['体重'].plot(kind='box')
BSdata['体重'].plot(kind='density',title='Density')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(1,3),kind='box')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(1,3),kind='density')
BSdata[['身高','体重','支出']].plot(subplots=True,layout=(3,1),kind='density')

T1=BSdata['开设'].value_counts();T1
pd.DataFrame({'频数':T1,'频率':T1/T1.sum()*100})
T1.plot(kind='bar'); #T1.sort_values().plot(kind='bar')
T1.plot(kind='pie')


######## 05.pdf #########lect14
## python数据可视化分析 ##

#### 特殊统计图的绘制 ####

#初等函数
import math
import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(0,2*math.pi);x
#fig,ax=plt.subplots(2,2,figsize=(15,12))
plt.plot(x,np.sin(x))
plt.plot(x,np.cos(x))
plt.plot(x,np.log(x))    #log(x)图像结果？？？
plt.plot(x,np.exp(x))

#极坐标(加公式)
t=np.linspace(0,2*math.pi)
x=2*np.sin(t)
y=3*np.cos(t)
plt.plot(x,y)
plt.text(0,0,r'$\frac{x^2}{2}+\frac{y^2}{3}=1$',fontsize=15)  #加公式

#气泡图
import pandas as pd
BSdata = pd.read_csv('BSdata.csv',encoding='utf-8')
plt.scatter(BSdata['身高'], BSdata['体重'], s=BSdata['支出'])   #s=:气泡的大小

#三维曲面图
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)      #生成三维坐标轴
X=np.linspace(-4,4,21) #X = np.arange(-4, 4, 0.5);
Y=np.linspace(-4,4,21) #Y = np.arange(-4, 4, 0.5)
X,Y = np.meshgrid(X, Y)   #X.shape：（21,21）
#使用meshgrid方法，你只需要构造一个表示x轴上的坐标的向量和一个表示y轴上的坐标的向量;然后作为参数给到meshgrid(),该函数就会返回相应维度的网格点矩阵;
Z = np.sqrt(X**2 + Y**2)
ax.plot_surface(X, Y, Z)     #plot_surface()

#三维散点图
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(BSdata['身高'], BSdata['体重'], BSdata['支出'])


##### seaborn统计绘图 #######
import seaborn as sns 

### 常用统计图

#箱线图 boxplot
# 绘制箱线图
sns.boxplot(x=BSdata['身高'])
# 竖着放的箱线图，也就是将x换成y
sns.boxplot(y=BSdata['身高'])

# 分组绘制箱线图，分组因子是性别，在x轴不同位置绘制
sns.boxplot(x='性别', y='身高',data=BSdata)

sns.boxplot(x='开设', y='支出',hue='性别',data=BSdata)

#小提琴图 violinplot
sns.violinplot(x='性别', y='身高',data=BSdata)
sns.violinplot(x='开设', y='支出',hue='性别',data=BSdata)

#点图 stripplot
sns.stripplot(x='性别', y='身高',data=BSdata)
sns.stripplot(x='性别', y='身高',data=BSdata,jitter=True)
sns.stripplot(y='性别', x='身高',data=BSdata,jitter=True)

#条图 barplot
sns.barplot(x='性别', y='身高',data=BSdata,ci=0,palette="Blues_d")

#计数图 countplot
sns.countplot(x='性别',data=BSdata)
sns.countplot(y='开设',data=BSdata)
sns.countplot(x='性别',hue='开设',data=BSdata)

#分组关系图 catplot
sns.catplot(x='性别',col='开设', col_wrap=3,data=BSdata,kind='count', height=2.5, aspect=.8)
##以col=为分组因子

#概率分布图

#单变量：distplot
sns.distplot(BSdata['身高'], kde=True, bins=20, rug=True);
# kde控制是否画kde曲线，bins是分组数，rug控制是否画样本点

#双变量：jointplot
sns.jointplot(x='身高', y='体重', data=BSdata);

#多变量：pairplot
sns.pairplot(BSdata[['身高','体重','支出']]);
#默认对角线为直方图 （histgram），非对角线为散点图


####### ggplot绘图系统 #######

## qplot函数 ##

from ggplot import *
import matplotlib.pyplot as plt #基本绘图包
plt.rcParams['font.sans-serif']=['KaiTi']; #SimHei黑体
plt.rcParams['axes.unicode_minus']=False; #正常显示图中负号

#直方图
qplot('身高',data=BSdata, geom='histogram')

#条形图
qplot('开设',data=BSdata, geom='bar')

#散点图
qplot('身高','体重',data=BSdata,color='性别')

## ggplot基本绘图 ##

#图层/添加图层：+
GP=ggplot(aes(x='身高',y='体重'),data=BSdata);GP #绘制直角坐标系
GP+geom_point()+geom_line()

#直方图
ggplot(BSdata,aes(x='身高'))+ geom_histogram()

#散点图
ggplot(BSdata,aes(x='身高',y='体重')) + geom_point()

#线图
ggplot(BSdata,aes(x='支出'))+geom_line(aes(y='身高'))

#分面图/使用 facet_wrap 参数可以按照类型绘制分面图
ggplot(BSdata,aes(x='身高',y='体重')) + geom_point() + facet_wrap('性别')

ggplot(BSdata,aes(x='身高',y='体重',color='性别'))+geom_point()+theme_bw()   #theme_bw() 默认为白色背景的主题



