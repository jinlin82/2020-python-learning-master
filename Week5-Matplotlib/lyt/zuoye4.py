### zuoye4 ###

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['KaiTi'] #SimHei黑体
plt.rcParams['axes.unicode_minus']=False

###  1. 利用mtcars数据集，使用图形参数的方法作一个包含3行2列的面板图，要求第一列包含4，6，8缸汽车的mpg~wt 的散点图，第二列包含包含4，6，8缸汽车的mpg~hp 的散点图, 并对子图形的坐标轴范围、标签、边距等进行设置。
  
mtcars = pd.read_csv('mtcars.csv')
mtcars.head()

mtcars_4 = mtcars[mtcars.cyl==4]
mtcars_6 = mtcars[mtcars.cyl==6]
mtcars_8 = mtcars[mtcars.cyl==8]

fig,ax = plt.subplots(3,2,figsize=(8,10))

ax[0,0].scatter(x='mpg',y='wt',data=mtcars_4);ax[0,0].set_xlabel('mpg');ax[0,0].set_ylabel('wt');ax[0,0].set_title('4 缸汽车')
ax[1,0].scatter(x='mpg',y='wt',data=mtcars_6);ax[1,0].set_xlabel('mpg');ax[1,0].set_ylabel('wt');ax[1,0].set_title('6 缸汽车')
ax[2,0].scatter(x='mpg',y='wt',data=mtcars_8);ax[2,0].set_xlabel('mpg');ax[2,0].set_ylabel('wt');ax[2,0].set_title('8 缸汽车')
ax[0,1].scatter(x='mpg',y='hp',data=mtcars_4);ax[0,1].set_xlabel('mpg');ax[0,1].set_ylabel('wt');ax[0,1].set_title('4 缸汽车')
ax[1,1].scatter(x='mpg',y='hp',data=mtcars_6);ax[1,1].set_xlabel('mpg');ax[1,1].set_ylabel('wt');ax[1,1].set_title('6 缸汽车')
ax[2,1].scatter(x='mpg',y='hp',data=mtcars_8);ax[2,1].set_xlabel('mpg');ax[2,1].set_ylabel('wt');ax[2,1].set_title('8 缸汽车')

fig.tight_layout()


###  2. 利用trees 数据集，完成面板图，要求：
     ### 1. 作 Volume~Girth 的散点图，要求y轴在右边，在图像下方给出Girth的箱线图
     ### 2. 作 Volume~Height 的散点图，要求y轴在左边，在图像下方给出Height的箱线图
     ### 3. 在两个散点图中间给出 Volume 的箱线图.
     ### 4. 对子图形的坐标轴范围、标签、边距等进行设置

trees = pd.read_csv('trees.csv')
trees.head()

fig = plt.figure(constrained_layout=True,figsize=(15,8))
gs = fig.add_gridspec(3,5)   
## add_gridspec():用于调整各个ax的所占面积比例

ax1 = fig.add_subplot(gs[:-1,0:2])
ax1.yaxis.set_ticks_position('right')     # y轴在右边
ax1.set_xlabel('Girth');ax1.set_ylabel('Volume')   
ax1.scatter('Girth','Volume',data=trees)

ax2 = fig.add_subplot(gs[:-1,2])
ax2.boxplot(trees.Volume,patch_artist=True,widths=0.5)
ax2.set_title('Volume_boxplot')

ax3 = fig.add_subplot(gs[:-1,-2:])
ax3.yaxis.set_ticks_position('left')     # y轴在左边
ax3.set_xlabel('Volume');ax3.set_ylabel('Height')
ax3.scatter('Volume','Height',data=trees)

ax4 = fig.add_subplot(gs[-1,0:2])
ax4.boxplot(trees.Girth,vert=False,patch_artist=True,widths=0.5)     
# vert=False:箱线图水平摆放; 
# patch_artist=True:填充箱体的颜色; 
# widths：指定箱线图的宽度
ax4.set_title('Girth_boxplot')

ax5 = fig.add_subplot(gs[-1,-2:])
ax5.boxplot(trees.Height,vert=False,patch_artist=True,widths=0.5)    
ax5.set_title('Height_boxplot')

fig.tight_layout()


###  3. 在同一个图形中画出以下函数的曲线：
     ### 1. y=x^2+2*x+3
     ### 2. y=x^3+2*x^2+3*x+4
     ### 3. y=3+2*ln(x)
     ### 4. y=1+exp(x)
     ### 5. y=10-3*0.7^x
     ### 6. y=3*x+4
	 ### 7. 对以上线型添加latex公式图例
     ### 7. y=3
     ### 8. x=1
     ### 9. 要求设置为不同的线型和颜色
     ### 10. 添加曲线的图例(二次曲线，三次曲线，对数曲线等)
     ### 11. 在图形中添加 两个点， 一条线段，一个矩形，一个说明文字
     ### 12. 把整个图形的背景设置为黄色

x = np.linspace(-3,3,61)
y1 = x**2+2*x+3
y2 = x**3+2*x**2+3*x+4
y3 = 3+2*np.log(x)       #自然对数
y4 = 1+np.exp(x)
y5 = 10-3*0.7**x
y6 = 3*x+4

fig,ax = plt.subplots(figsize=(6,6))

plt.plot(x,y1,'b-',label=u'$y=x^2+2*x+3$二次曲线')
plt.plot(x,y2,'r+-',label=u'$y=x^3+2*x^2+3*x+4$三次曲线')
plt.plot(x,y3,'g--',label=u'$y=3+2*ln(x)$对数曲线')
plt.plot(x,y4,'k.',label=u'$y=1+exp(x)$指数曲线')
plt.plot(x,y5,'k.-',label=u'$y=10-3*0.7^x$指数曲线')
plt.plot(x,y6,'c-',label=u'$y=3*x+4$一次曲线')

plt.axvline(x=1)
plt.axhline(y=3)

plt.text(-1,-8,'. .',size=40)    # 点
plt.axvline(x=-0.5, ymin=0.05, ymax=0.2,c='k')    # 线段
plt.text(1.8,-10,'       ',bbox=dict(alpha=0.8))  # 矩形
plt.text(2,-5,'矩阵',size=20)   # 说明文字

ax.set_facecolor('yellow')    # 设置背景色

plt.legend()
plt.show()
 

###  4. 导入GDP数据集，分别作：
     ### 1. CPI向量的点图
     ### 2. 以Kapital为自变量，GDP为因变量，作它们的散点图
     ### 3. 作GDP数据集的散点图矩阵
     ### 4. 根据需要设置以上图形的点型，颜色，坐标轴范围，标题等选项

GDP = pd.read_csv('GDP.csv')
GDP.head()

## 题1 ##
CPI = GDP.iloc[:,-1]
plt.plot(CPI,'go');plt.ylabel('CPI')
plt.title('CPI点图')
plt.show()

## 题2 ##
plt.scatter('Kapital','GDP',data=GDP)
plt.xlabel('Kapital');plt.ylabel('GDP')
plt.title('Kapital-GDP散点图')
plt.show()

## 题3 ##
import seaborn as sns
sns.pairplot(GDP,diag_kind='kde',kind='scatter')
plt.title('散点图矩阵')
plt.show()
