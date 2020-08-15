#1. 利用mtcars数据集，使用图形参数的方法作一个包含3行2列的面板图，要求第一列包含4，6，8缸汽车的mpg~wt 的散点图，第二列包含包含4，6，8缸汽车的mpg~hp 的散点图, 并对子图形的坐标轴范围、标签、边距等进行设置。
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import math
plt.rcParams['font.sans-serif']=['SimHei']##SimHei黑体
plt.rcParams['axes.unicode_minus']=False##正常显示图中负号

mtcars=pd.read_csv('./data4/mtcars.csv')

plt.figure(figsize=(12,15))##图形大小
cyl4=mtcars[mtcars['cyl']==4]
cyl6=mtcars[mtcars['cyl']==6]
cyl8=mtcars[mtcars['cyl']==8]
plt.subplot(321),plt.scatter('mpg','wt',data=cyl4)
plt.xlabel('mpg');plt.ylabel('wt');plt.title("4缸汽车的mpg-wt散点图")
plt.xlim(20,36);plt.ylim(1,3.5)
plt.subplot(322),plt.scatter('mpg','hp',data=cyl4)
plt.xlabel('mpg');plt.ylabel('hp');plt.title("4缸汽车的mpg-hp散点图")
plt.subplot(323),plt.scatter('mpg','wt',data=cyl6)
plt.xlabel('mpg');plt.ylabel('wt');plt.title("6缸汽车的mpg-wt散点图")
plt.subplot(324),plt.scatter('mpg','hp',data=cyl6)
plt.xlabel('mpg');plt.ylabel('hp');plt.title("6缸汽车的mpg-hp散点图")
plt.subplot(325),plt.scatter('mpg','wt',data=cyl8)
plt.xlabel('mpg');plt.ylabel('wt');plt.title("8缸汽车的mpg-wt散点图")
plt.subplot(326),plt.scatter('mpg','hp',data=cyl8)
plt.xlabel('mpg');plt.ylabel('hp');plt.title("8缸汽车的mpg-hp散点图")

##使用agg/apply画图？？
cyl4=mtcars[mtcars['cyl']==4]
cyl6=mtcars[mtcars['cyl']==6]
cyl8=mtcars[mtcars['cyl']==8]
cyl=[cyl4,cyl6,cyl8]
cyl[0]
cyl[1]
cyl[2]
fig,ax=plt.subplots(3,2,figsize=(12,15))
for i in range(len(cyl)):
      ax[i][0].scatter('mpg','wt',data=cyl[i])
      ax[i][0].set_xlabel('mpg');ax[i][0].set_ylabel('wt')
      ax[i][1].scatter('mpg','hp',data=cyl[i])
      ax[i][1].set_xlabel('mpg');ax[i][1].set_ylabel('hp')
      if i==0:
         ax[i][0].set_title("4缸汽车的mpg-wt散点图")
         ax[i][1].set_title("4缸汽车的mpg-hp散点图")
      if i==1:
         ax[i][0].set_title("6缸汽车的mpg-wt散点图")
         ax[i][1].set_title("6缸汽车的mpg-hp散点图")
      if i==2:
         ax[i][0].set_title("8缸汽车的mpg-wt散点图")
         ax[i][1].set_title("8缸汽车的mpg-hp散点图")
   


# 2. 利用trees 数据集，完成面板图，要求：
# 1. 作 Volume~Girth 的散点图，要求y轴在右边，在图像下方给出Girth的箱线图
#   2. 作 Volume~Height 的散点图，要求y轴在左边，在图像下方给出Height的箱线图
#   3. 在两个散点图中间给出 Volume 的箱线图.

trees=pd.read_csv('./data4/trees.csv')

plt.figure(figsize=(12,10))
plt.subplot(231),plt.scatter(trees['Volume'],trees['Girth'])
plt.gca().yaxis.set_ticks_position('right')
plt.title("Volume-Girth的散点图");plt.xlabel('Volume');plt.ylabel('Girth')
plt.subplot(234),sns.boxplot(y='Girth',data=trees)
plt.title("Girth的箱线图");plt.ylabel('Girth')
plt.subplot(233),plt.scatter(trees['Volume'],trees['Height'])
plt.title("Volume-Height的散点图");plt.xlabel('Volume');plt.ylabel('Height')
plt.subplot(236),sns.boxplot(y='Height',data=trees)
plt.title("Height的箱线图");plt.ylabel('Height')
plt.subplot(232),sns.boxplot(y='Volume',data=trees)
plt.title("Volume的箱线图");plt.ylabel('Volume')
 
##调整图形和画布大小
# fig=plt.figure(constrained_layout=True,figsize=(5,5))
# gs=fig.add_gridspec(3,3)
# f_ax1=fig.add_subplot(gs[0,:])
# f_ax1.set_title('gs[0,:]')
# f_ax2=fig.add_subplot(gs[1,:-1])
# f_ax2.set_title('gs[1,:-1]')
# f_ax3=fig.add_subplot(gs[1:,-1])
# f_ax3.set_title('gs[1:,-1]')
# f_ax4=fig.add_subplot(gs[-1,0])
# f_ax4.set_title('gs[-1,0]')
# f_ax5=fig.add_subplot(gs[-1,-2])
# f_ax5.set_title('gs[-1,-2]')
 

fig=plt.figure(constrained_layout=True,figsize=(9,6))
gs=fig.add_gridspec(4,5)
##1
f_ax1=fig.add_subplot(gs[:-1,0:2])
plt.scatter(trees['Volume'],trees['Girth'])
plt.gca().yaxis.set_ticks_position('right')
plt.title("Volume-Girth的散点图");plt.xlabel('Volume');plt.ylabel('Girth')
##2
f_ax2=fig.add_subplot(gs[:-1,2])
sns.boxplot(y='Volume',data=trees)
plt.title("Volume的箱线图");plt.ylabel('Volume')
##3
f_ax3=fig.add_subplot(gs[:-1,3:])
plt.scatter(trees['Volume'],trees['Height'])
plt.title("Volume-Height的散点图");plt.xlabel('Volume');plt.ylabel('Height')
##4
f_ax4=fig.add_subplot(gs[-1:,:2])
sns.boxplot(x='Girth',data=trees)
plt.title("Girth的箱线图");plt.xlabel('Girth')
##5
f_ax5=fig.add_subplot(gs[-1:,-2:])
sns.boxplot(x='Height',data=trees)
plt.title("Height的箱线图");plt.xlabel('Height')



   #   4. 对子图形的坐标轴范围、标签、边距等进行设置
#   3. 在同一个图形中画出以下函数的曲线：
   #   1. y=x^2+2*x+3
   #   2. y=x^3+2*x^2+3*x+4
   #   3. y=3+2*ln(x)
   #   4. y=1+exp(x)
   #   5. y=10-3*0.7^x
   #   6. y=3*x+4
	#    7. 对以上线型添加latex公式图例
   #   7. y=3
   #   8. x=1
   #   9. 要求设置为不同的线型和颜色
   #   10. 添加曲线的图例(二次曲线，三次曲线，对数曲线等)
   #   11. 在图形中添加 两个点， 一条线段，一个矩形，一个说明文字
   #   12. 把整个图形的背景设置为黄色

fig,ax=plt.subplots(1,1,figsize=(10,6))
x=np.linspace(-10,10,100)
plt.xlim(-10,10)
plt.ylim(-20,20)
plt.plot(x,x**2+2*x+3,'g--',label=r'$y=x^2+2x+3$二次曲线')
plt.plot(x,x**3+2*x**2+3*x+4,'r*',label=r'$y=x^3+2x^2+3x+4$三次曲线')
plt.plot(x,3+2*np.log(x),'b-',label=r'$y=3+2ln(x)$对数曲线')##log
plt.plot(x,1+np.exp(x),'y+',label=r'$y=1+exp(x)$指数曲线')
plt.plot(x,10-3*0.7**x,'g^',label=r'$y=10-3*0.7^x$指数曲线')
plt.plot(x,3*x+4,'r-',label=r'$y=3x+4$直线')
plt.axvline(x=1)
plt.axhline(y=3)
plt.axhspan(ymin=-10,ymax=-6,alpha=1,color='pink')##alpha表示颜色深浅
plt.axvspan(xmin=-8,xmax=-6,alpha=0.6,color='blue')
plt.text(0,0,'原点')
plt.legend()
ax.set_facecolor('yellow')##图片底色
fig.set_facecolor('yellow')##画布底色

 
#   4. 导入GDP数据集，分别作：
gdp=pd.read_csv('./data4/GDP.csv')

#      1. CPI向量的点图
plt.figure(figsize=(6,4))
sns.stripplot('CPI',data=gdp,color='red')
plt.xlim(0.9,1.3)
plt.xlabel('CPI')
plt.title('CPI的点图')

#      2. 以Kapital为自变量，GDP为因变量，作它们的散点图
plt.scatter('Kapital','GDP',data=gdp,color='green')
plt.xlim(1,180000);plt.ylim(0,450000)
plt.xlabel('Kapital')
plt.ylabel('GDP')
plt.title('Kapital-GDP的散点图')

#      3. 作GDP数据集的散点图矩阵
gdp.columns
gdp
sns.pairplot(gdp[['Year','GDP','GDPRealRate','Labor','Kapital','KR','Technology','Energy','HR','CPI']])
plt.title("GDP数据集的散点图矩阵")


#      4. 根据需要设置以上图形的点型，颜色，坐标轴范围，标题等选项

