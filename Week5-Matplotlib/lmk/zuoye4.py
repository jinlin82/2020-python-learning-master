# 1. 利用mtcars数据集，使用图形参数的方法作一个包含3行2列的面板图，要求第一列包
#    含4，6，8缸汽车的mpg~wt的散点图，第二列包含包含4，6，8缸汽车的mpg~hp的散
#    点图, 并对子图形的坐标轴范围、标签、边距等进行设置。

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus']=False

mtcars=pd.read_csv('./data/mtcars.csv')
mt1=mtcars[mtcars['cyl']==4]
mt2=mtcars[mtcars['cyl']==6]
mt3=mtcars[mtcars['cyl']==8]

plt.figure(figsize=(10,9))
plt.subplots_adjust(left=0,bottom=0,right=1,top=1,hspace=0.1,wspace=0.1)
plt.subplot(321)
plt.scatter(mt1.wt,mt1.mpg)
plt.title('4缸汽车')
plt.xlabel('wt')
plt.ylabel('mpg')
plt.xticks(np.arange(1,6,0.5))
plt.yticks(np.arange(10,40,5))

plt.subplot(323)
plt.scatter(mt2.wt,mt2.mpg)
plt.title('6缸汽车')
plt.xlabel('wt')
plt.ylabel('mpg')
plt.xticks(np.arange(1,6,0.5))
plt.yticks(np.arange(10,40,5))

plt.subplot(325)
plt.scatter(mt3.wt,mt3.mpg)
plt.title('8缸汽车')
plt.xlabel('wt')
plt.ylabel('mpg')
plt.xticks(np.arange(1,6,0.5))
plt.yticks(np.arange(10,40,5))

plt.subplot(322)
plt.scatter(mt1.hp,mt1.mpg)
plt.title('4缸汽车')
plt.xlabel('hp')
plt.ylabel('mpg')
plt.xticks(np.arange(50,400,50))
plt.yticks(np.arange(10,40,5))

plt.subplot(324)
plt.scatter(mt2.hp,mt2.mpg)
plt.title('6缸汽车')
plt.xlabel('hp')
plt.ylabel('mpg')
plt.xticks(np.arange(50,400,50))
plt.yticks(np.arange(10,40,5))

plt.subplot(326)
plt.scatter(mt3.hp,mt3.mpg)
plt.title('8缸汽车')
plt.xlabel('hp')
plt.ylabel('mpg')
plt.xticks(np.arange(50,400,50))
plt.yticks(np.arange(10,40,5))

plt.tight_layout()
plt.show()

### 用agg/apply画多图
fig,axes=plt.subplots(3,2,figsize=(10,9))
def plotax(i):
    if i%2 ==0:
        axes[i].set_xlabel('wt')
        axes[i].set_ylabel('mpg')
        axes[i].set_xticks(np.arange(1,6,0.5))
        axes[i].set_yticks(np.arange(10,40,5))
    else:
        axes[i].set_xlabel('hp')
        axes[i].set_ylabel('mpg')
        axes[i].set_xticks(np.arange(50,400,50))
        axes[i].set_yticks(np.arange(10,40,5))
def plottitle1(i):
    if i==1|2:
        axes[i].set_title('4缸汽车')
def plottitle2(i):
    if i==3|4:
        axes[i].set_title('6缸汽车')
def plottitle3(i):
    if i==5|6:
        axes[i].set_title('8缸汽车')

pd.DataFrame(axes[0]).agg([plotax,plottitle1])

fig,axes=plt.subplots(3,2,figsize=(10,9))
axes[0].scatter(mt1.wt,mt1.mpg)
axes[0].set_xlabel('wt')
axes[0].set_ylabel('mpg')
axes[0].set_xticks(np.arange(1,6,0.5))
axes[0].set_yticks(np.arange(10,40,5))
plt.show()

# 2. 利用trees数据集，完成面板图，要求：
## 1. 作 Volume~Girth 的散点图，要求y轴在右边，在图像下方给出Girth的箱线图
## 2. 作 Volume~Height 的散点图，要求y轴在左边，在图像下方给出Height的箱线图
## 3. 在两个散点图中间给出 Volume 的箱线图
## 4. 对子图形的坐标轴范围、标签、边距等进行设置

import seaborn as sns
trees=pd.read_csv('./data/trees.csv')
plt.figure(figsize=(12,8))
plt.subplots_adjust(left=0,bottom=0,right=1,top=1,hspace=0.1,wspace=0.1)

plt.subplot(231)
plt.scatter(trees.Girth,trees.Volume)
plt.title('Volume-Girth')
plt.xlabel('Girth')
plt.ylabel('Volume')
plt.xticks(np.arange(8,22,1))
plt.yticks(np.arange(10,85,5))
ax=plt.gca()
ax.yaxis.set_ticks_position('right')

plt.subplot(233)
plt.scatter(trees.Height,trees.Volume)
plt.title('Volume-Height')
plt.xlabel('Height')
plt.ylabel('Volume')
plt.xticks(np.arange(60,95,5))
plt.yticks(np.arange(10,85,5))

plt.subplot(232)
sns.boxplot(y=trees.Volume)
plt.title('Volume boxplot')

plt.subplot(234)
sns.boxplot(x=trees.Girth)
plt.title('Girth boxplot')

plt.subplot(236)
sns.boxplot(x=trees.Height)
plt.title('Height boxplot')

plt.tight_layout()
plt.show()

### 改变子图的布局和大小
fig=plt.figure(constrained_layout=True,figsize=(12,8))
gs=fig.add_gridspec(3,5)

f_ax1=fig.add_subplot(gs[:-1,0:2])
plt.scatter(trees.Girth,trees.Volume)
plt.title('Volume-Girth')
plt.xlabel('Girth')
plt.ylabel('Volume')
plt.xticks(np.arange(8,22,1))
plt.yticks(np.arange(10,85,5))
ax=plt.gca()
ax.yaxis.set_ticks_position('right')

f_ax2=fig.add_subplot(gs[:-1,2])
sns.boxplot(y=trees.Volume)
plt.title('Volume boxplot')

f_ax3=fig.add_subplot(gs[:-1,-2:])
plt.scatter(trees.Height,trees.Volume)
plt.title('Volume-Height')
plt.xlabel('Height')
plt.ylabel('Volume')
plt.xticks(np.arange(60,95,5))
plt.yticks(np.arange(10,85,5))

f_ax4=fig.add_subplot(gs[-1:,0:2])
sns.boxplot(x=trees.Girth)
plt.title('Girth boxplot')

f_ax5=fig.add_subplot(gs[-1:,-2:])
sns.boxplot(x=trees.Height)
plt.title('Height boxplot')

plt.show()

# 3. 在同一个图形中画出以下函数的曲线：
## 1. y=x^2+2*x+3
## 2. y=x^3+2*x^2+3*x+4
## 3. y=3+2*ln(x)
## 4. y=1+exp(x)
## 5. y=10-3*0.7^x
## 6. y=3*x+4
### 对以上线型添加latex公式图例
## 7. y=3
## 8. x=1
## 9. 要求设置为不同的线型和颜色
## 10. 添加曲线的图例(二次曲线，三次曲线，对数曲线等)
## 11. 在图形中添加 两个点， 一条线段，一个矩形，一个说明文字
## 12. 把整个图形的背景设置为黄色

import math
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(111)
ax.set_facecolor('yellow')
x1=np.arange(-2,5,0.01)
plt.plot(x1,x1**2+2*x1+3,'-b')
plt.text(5,38,r'$y=x^{2}+2x+3$')

plt.plot(x1,x1**3+2*x1**2+3*x1+4,'--g')
plt.text(5,190,r'$y=x^{3}+2x^{2}+3x+4$')

x2=np.arange(0.01,5,0.01)
y2=[3+2*math.log10(a) for a in x2]
plt.plot(x2,y2,'or')
plt.text(5,4,r'$y=3+2\ln x$')

plt.plot(x1,1+math.e**x1,'^k')
plt.text(5,1+math.e**(5),r'$y=1+e^{x}$')

plt.plot(x1,10-3*0.7**x1,'-.c')
plt.text(5,10,r'$y=10-3\times0.7^{x}$')

plt.plot(x1,3*x1+4,':m')
plt.text(5,19,r'$3x+4$')

plt.axhline(y=3)
plt.axvline(x=1)

plt.legend(['二次曲线','三次曲线','对数曲线','exp曲线','指数曲线','线性函数'],fontsize='x-large')

plt.plot([1,25],[2,50],'w')
rectangle=plt.Rectangle((1,50),1,25,color='w')
ax.add_patch(rectangle)
plt.annotate('多线图',xy=(10,100),xytext=(10,120),color='r',size=30)

plt.show()

# 4. 导入GDP数据集，分别作：
## 1. CPI向量的点图

gdp=pd.read_csv('./data/GDP.csv')
sns.stripplot(x='CPI',data=gdp,color='m')

## 2. 以Kapital为自变量，GDP为因变量，作它们的散点图

plt.scatter(gdp.Kapital,gdp.GDP,marker='+')
plt.title('GDP-Kapital')
plt.xlabel('Kapital')
plt.ylabel('GDP')
plt.xticks(np.arange(0,225000,25000))
plt.yticks(np.arange(0,500000,50000))
plt.show()

## 3. 作GDP数据集的散点图矩阵

sns.pairplot(gdp)

## 4. 根据需要设置以上图形的点型，颜色，坐标轴范围，标题等选项