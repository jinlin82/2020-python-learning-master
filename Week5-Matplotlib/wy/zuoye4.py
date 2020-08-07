## 作业内容 ##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimSun']

# 1. 利用mtcars数据集，使用图形参数的方法作一个包含3行2列的面板图，要求第一列包含4，6，8缸汽车的mpg~wt 的散点图，第二列包含包含4，6，8缸汽车的mpg~hp 的散点图, 并对子图形的坐标轴范围、标签、边距等进行设置。
mtcars=pd.read_csv('./data/mtcars.csv')
cyl1=mtcars[mtcars.cyl==4]
cyl2=mtcars[mtcars.cyl==6]
cyl3=mtcars[mtcars.cyl==8]
fig1=plt.figure(figsize=(12,16))
plt.subplot(321)
plt.scatter(cyl1.wt,cyl1.mpg)
plt.ylim([20,35])
plt.xlim([1,4])
plt.xlabel('wt')
plt.ylabel('mpg')
plt.title('四缸')

plt.subplot(322)
plt.scatter(cyl1.hp,cyl1.mpg)
plt.ylim([20,35])
plt.xlim([50,120])
plt.xlabel('hp')
plt.ylabel('mpg')
plt.title('四缸')

plt.subplot(323)
plt.scatter(cyl2.wt,cyl2.mpg)
plt.ylim([18,22])
plt.xlim([1,4])
plt.xlabel('wt')
plt.ylabel('mpg')
plt.title('六缸')

plt.subplot(324)
plt.scatter(cyl2.hp,cyl2.mpg)
plt.ylim([18,22])
plt.xlim([100,180])
plt.xlabel('hp')
plt.ylabel('mpg')
plt.title('六缸')

plt.subplot(325)
plt.scatter(cyl3.wt,cyl3.mpg)
plt.ylim([10,20])
plt.xlim([3,6])
plt.xlabel('wt')
plt.ylabel('mpg')
plt.title('八缸')

plt.subplot(326)
plt.scatter(cyl3.hp,cyl3.mpg)
plt.ylim([10,20])
plt.xlim([150,350])
plt.xlabel('hp')
plt.ylabel('mpg')
plt.title('八缸')

# 2. 利用trees 数据集，完成面板图，要求：
trees=pd.read_csv('./data/trees.csv')
  ## 1. 作 Volume~Girth 的散点图，要求y轴在右边，在图像下方给出Girth的箱线图
  ## 2. 作 Volume~Height 的散点图，要求y轴在左边，在图像下方给出Height的箱线图
  ## 3. 在两个散点图中间给出 Volume 的箱线图.
  ## 4. 对子图形的坐标轴范围、标签、边距等进行设置
fig2=plt.figure(constrained_layout=True,figsize=(15,10))
gs=fig2.add_gridspec(4,5)

f1=fig2.add_subplot(gs[:-1,0:2])
plt.scatter(trees.Girth,trees.Volume)
plt.xlabel('Girth')
plt.ylabel('Volume')
xy=plt.gca()
xy.yaxis.set_ticks_position('right')

f2=fig2.add_subplot(gs[:-1,2])
sns.boxplot(y=trees.Volume)
plt.title('Volume箱线图')

f3=fig2.add_subplot(gs[:-1,3:])
plt.scatter(trees.Height,trees.Volume)
plt.xlabel('Height')
plt.ylabel('Volume')

f4=fig2.add_subplot(gs[-1,0:2])
sns.boxplot(x=trees.Girth)
plt.title('Girth箱线图')

f5=fig2.add_subplot(gs[-1,3:])
sns.boxplot(x=trees.Height)
plt.title('Height箱线图')

# 3. 在同一个图形中画出以下函数的曲线：
  ## 1. y=x^2+2*x+3
  ## 2. y=x^3+2*x^2+3*x+4
  ## 3. y=3+2*ln(x)
  ## 4. y=1+exp(x)
  ## 5. y=10-3*0.7^x
  ## 6. y=3*x+4
  ##   7. 对以上线型添加latex公式图例
  ## 7. y=3
  ## 8. x=1
  ## 9. 要求设置为不同的线型和颜色
  ## 10. 添加曲线的图例(二次曲线，三次曲线，对数曲线等)
  ## 11. 在图形中添加 两个点， 一条线段，一个矩形，一个说明文字
  ## 12. 把整个图形的背景设置为黄色
plt.rcParams['axes.unicode_minus']=False
x=np.linspace(-0.5,2.5,201);x
fig3=plt.figure(figsize=(6,6))
ax=fig3.add_subplot(111)
ax.set_facecolor('yellow')
plt.plot(x,x**2+2*x+3,'r--',label=r'二次曲线')
ax.annotate(r'$y=x^2+2*x+3$',xy=(2,2**2+2*2+3),xytext=(1.5,15),arrowprops=dict(facecolor='black',shrink=0.05))
plt.plot(x,x**3+2*x**2+3*x+4,'k--',label=r'三次曲线')
ax.annotate(r'$y=x^3+2*x^2+3*x+4$',xy=(2,2**3+2*2**2+3*2+4),xytext=(1,35),arrowprops=dict(facecolor='black',shrink=0.05))
plt.plot(x,3+2*np.log(x),'g--',label=r'对数曲线')
plt.text(0.25,3+2*np.log(0.25),r'$y=3+2*ln(x)$')
plt.plot(x,1+np.exp(x),'black',label=u'指数曲线')
plt.text(1.5,1+np.exp(1.5),r'$y=1+exp(x)$')
plt.plot(x,10-3*0.7**x,'blue',label=u'指数曲线')
plt.text(-0.5,10-3*0.7**(-0.5),r'$y=10-3*0.7^x$')
plt.plot(x,3*x+4,'purple',label=r'一元函数')
ax.annotate(r'$y=3*x+4$',xy=(0.75,3*(0.75)+4),xytext=(-0.25,15),arrowprops=dict(facecolor='black',shrink=0.05))
plt.axvline(x=1)
plt.axhline(y=3)
plt.legend()

rect=plt.Rectangle((0.5,20),0.5,10,color='r')
ax.add_patch(rect)

plt.scatter(0.5,15,s=20,color='k')
plt.scatter(1.5,25,s=20,color='k')
ax.annotate('一个点',xy=(1.5,25),xytext=(1.5,30),arrowprops=dict(facecolor='black',shrink=0.01))
plt.plot([0.5,1.5],[15,25],color='g')

# 4. 导入GDP数据集，分别作：
gdp=pd.read_csv('./data/GDP.csv');gdp

  ## 1. CPI向量的点图
sns.stripplot(x='CPI',data=gdp,color='g')

  ## 2. 以Kapital为自变量，GDP为因变量，作它们的散点图
plt.scatter(gdp.Kapital,gdp.GDP,color='purple')
plt.xlabel('Kapital')
plt.ylabel('GDP')

  ## 3. 作GDP数据集的散点图矩阵
  del gdp['Year']
sns.pairplot(gdp)

  ## 4. 根据需要设置以上图形的点型，颜色，坐标轴范围，标题等选项

