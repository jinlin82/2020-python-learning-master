#1. 利用mtcars数据集，使用图形参数的方法作一个包含3行2列的面板图，要求第一列包含
#   4，6，8缸汽车的mpg~wt 的散点图，第二列包含包含4，6，8缸汽车的mpg~hp 的散点
#   图, 并对子图形的坐标轴范围、标签、边距等进行设置。

import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus']=False

cars = pd.read_csv('./data/mtcars.csv')
cars.head()

plt.figure(figsize=(20,30))
fig,axes = plt.subplots(3,2)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=2, hspace=2)

axes[0,0].scatter(cars[cars['cyl']==4]['mpg'],cars[cars['cyl']==4]['wt'])
axes[0,0].set_xlim(20,40)
axes[0,0].set_ylim(0,5)
axes[0,0].set_xlabel('mpg')
axes[0,0].set_ylabel('wt')
axes[0,0].set_title('4缸汽车')

axes[1,0].scatter(cars[cars['cyl']==6]['mpg'],cars[cars['cyl']==6]['wt'])
axes[1,0].set_xlim(15,25)
axes[1,0].set_ylim(2,4)
axes[1,0].set_xlabel('mpg')
axes[1,0].set_ylabel('wt')
axes[1,0].set_title('6缸汽车')

axes[2,0].scatter(cars[cars['cyl']==8]['mpg'],cars[cars['cyl']==8]['wt'])
axes[2,0].set_xlim(0,25)
axes[2,0].set_ylim(2,7)
axes[2,0].set_xlabel('mpg')
axes[2,0].set_ylabel('wt')
axes[2,0].set_title('8缸汽车')

axes[0,1].scatter(cars[cars['cyl']==4]['hp'],cars[cars['cyl']==4]['hp'])
axes[0,1].set_xlim(40,130)
axes[0,1].set_ylim(40,130)
axes[0,1].set_xlabel('mpg')
axes[0,1].set_ylabel('hp')
axes[0,1].set_title('4缸汽车')

axes[1,1].scatter(cars[cars['cyl']==6]['hp'],cars[cars['cyl']==6]['hp'])
axes[1,1].set_xlim(80,200)
axes[1,1].set_ylim(100,200)
axes[1,1].set_xlabel('mpg')
axes[1,1].set_ylabel('hp')
axes[1,1].set_title('6缸汽车')

axes[2,1].scatter(cars[cars['cyl']==8]['hp'],cars[cars['cyl']==8]['hp'])
axes[2,1].set_xlim(100,400)
axes[2,1].set_ylim(100,400)
axes[2,1].set_xlabel('mpg')
axes[2,1].set_ylabel('hp')
axes[2,1].set_title('8缸汽车')

#2. 利用trees 数据集，完成面板图，要求：
  ##1. 作 Volume~Girth 的散点图，要求y轴在右边，在图像下方给出Girth的箱线图
#f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
  ##2. 作 Volume~Height 的散点图，要求y轴在左边，在图像下方给出Height的箱线图
  ##3. 在两个散点图中间给出 Volume 的箱线图.

trees = pd.read_csv('./data/trees.csv')
trees.head()

fig = plt.figure(constrained_layout=True,figsize=(15,8))
gs = fig.add_gridspec(3,5)

##1

ax1 = fig.add_subplot(gs[0:2,0:2])
ax1.scatter(trees['Girth'],trees['Height'])
# 获取当前的坐标轴, gca = get current axis
#ax = plt.gca()
ax1.yaxis.set_ticks_position('right')
ax1.set_xlabel('Girth')
ax1.set_ylabel('Height') 

##2

ax2 = fig.add_subplot(gs[0:2,2])
ax2.boxplot(trees['Girth'],vert=True,patch_artist=True)
ax2.set_xlabel('Girth')

##3

ax3 = fig.add_subplot(gs[0:2,3:])
ax3.scatter(trees['Girth'],trees['Volume'])
ax3.yaxis.set_ticks_position('left')
ax3.set_xlabel('Girth')
ax3.set_ylabel('Volume') 

##4

ax4 = fig.add_subplot(gs[2,0:2])
ax4.boxplot(trees['Girth'],vert=False,patch_artist=True)
ax4.set_xlabel('Girth')

##5

ax5 = fig.add_subplot(gs[2,3:])
ax5.boxplot(trees['Girth'],vert=False,patch_artist=True)
ax5.set_xlabel('Girth')


#3. 在同一个图形中画出以下函数的曲线：
     ##1. y=x^2+2*x+3

     2. y=x^3+2*x^2+3*x+4
     3. y=3+2*ln(x)
     4. y=1+exp(x)
     5. y=10-3*0.7^x
     6. y=3*x+4
	   7. 对以上线型添加latex公式图例
     7. y=3
     8. x=1
     9. 要求设置为不同的线型和颜色
     10. 添加曲线的图例(二次曲线，三次曲线，对数曲线等)
     11. 在图形中添加 两个点， 一条线段，一个矩形，一个说明文字
     12. 把整个图形的背景设置为黄色

plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus']=False
x = np.linspace(-5,5,201)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_facecolor('yellow')
plt.plot(x,x**2+2*x+3,'b.-',label=u'二次曲线')
plt.text(4,4**2+2*4+3,r'$x^2+2*x+3$',size=15)

#ax.annotate(r'$y=x^2+2x+3$',xy=(5,5**2+5*2+3),xytext=(1.5,15),arrowprops=dict(facecolor='black',shrink=0.05))

plt.plot(x,3+2*np.log(x),'g--',label=r'对数曲线')
plt.text(2,3+2*np.log(2),r'$y=3+2ln(x)$',size=15)

plt.plot(x,1+np.exp(x),'k+-',label=u'指数曲线')
plt.text(0,1+np.exp(0),r'$y=3+2ln(x)$',size=15)

plt.plot(x,10-3*0.7**x,'rs-',label=u'指数曲线')
plt.text(-2,10-3*0.7**(-2),r'$10-3*0.7^x$',size=15)

plt.plot(x,3*x+4,'go',label=r'一次函数')
plt.text(-4,-4*0+4,u'$3x+4$',size=15)
#ax.annotate(r'$y=3*x+4$',xy=(0,3*0+4),xytext=(-2,10),arrowprops=dict(facecolor='black',shrink=0.05))

plt.axvline(x=1)
plt.axhline(y=3)

plt.text(-1,49,'. .',size=50) # 两个点
plt.axvline(x=0.5, ymin=0.3, ymax=1,c='k')  # 一条线段
rect=plt.Rectangle((0.5,20),0.5,10,color='r')
ax.add_patch(rect)#矩阵
plt.text(0.5,20,'矩阵',size=20)#说明文字

plt.legend()

#4. 导入GDP数据集，分别作：
    ## 1. CPI向量的点图

gdp = pd.read_csv('./data/GDP.csv')
gdp.head()
plt.plot(gdp['CPI'],'bo')
plt.ylabel('CPI')
plt.title('CPI点图')

     ##2. 以Kapital为自变量，GDP为因变量，作它们的散点图

plt.scatter(gdp['Kapital'],gdp['GDP'])
plt.xlabel('Kapital')
plt.ylabel('GDP')
plt.title('Kapital与GDP散点图')

     ##3. 作GDP数据集的散点图矩阵

gdp1 = gdp.iloc[:,1:]
gdp1.head()
sns.pairplot(data=gdp1)
#fig.savefig('./scatter_matrix.jpg')

     ##4. 根据需要设置以上图形的点型，颜色，坐标轴范围，标题等选项


#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111)

sns.pairplot(data=gdp1,vars=['GDP','Kapital'],palette='husl',kind='reg',diag_kind='kde',markers='s')
ax = plt.axes()
ax.set_xlim(0,10)
ax.set_ylim(0,10)
plt.suptitle('GDP')