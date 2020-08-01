import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math
plt.rcParams['font.sans-serif']=['KaiTi']
x=np.linspace(-4*math.pi,4*math.pi,100);x
plt.plot(x,np.sin(x))
plt.plot(x,np.cos(x))
plt.plot(x,np.log(x))
##代码一起运行则绘制在一张图上
##给图表加名称标注
plt.text(6,1,'$y=\log(x)+\sum_{i=1}^n x_i$')

BSdata=pd.read_csv('./data/BSdata.csv')
##气泡图
plt.scatter(BSdata.身高,BSdata.体重,s=BSdata.支出)

##三维图
from mpl_toolkits.mplot3d import Axes3D

X=np.linspace(-4,4,20) #X = np.arange(-4, 4, 0.5);
Y=np.linspace(-4,4,20) #Y = np.arange(-4, 4, 0.5)
X, Y = np.meshgrid(X, Y)
Z = np.sqrt(X**2 + Y**2)
fig1=plt.figure()
ax = Axes3D(fig1)
ax.plot_surface(X, Y, Z)
##形成交互式可旋转——右键—在终端中打开运行行/列 然后输入plt.show()

fig2=plt.figure()
ax = Axes3D(fig2)
# ax.scatter(X, Y, Z)
ax.scatter(BSdata.身高,BSdata.体重,BSdata.支出)##三维散点图
ax.scatter(BSdata.身高,BSdata.体重,BSdata.支出,s=50*np.random.rand(52))##三维气泡图

##seaborn作图
import seaborn as sns

#箱线图
sns.boxplot(BSdata.身高)##横向
plt.boxplot(BSdata.身高)##纵向

sns.boxplot(BSdata.身高)
sns.boxplot(x=BSdata.性别,y=BSdata.身高)

sns.boxplot(x=BSdata.开设,y=BSdata.支出,hue=BSdata.性别)
sns.boxplot(y=BSdata.开设,x=BSdata.支出,hue=BSdata.性别)
plt.text(80,2,'abc')
plt.text(80,1,r'$\bar x$')

# 绘制箱线图
sns.boxplot(x=BSdata['身高'])
# 竖着放的箱线图，也就是将x换成y
sns.boxplot(y=BSdata['身高'])
# 分组绘制箱线图，分组因子是性别，在x轴不同位置绘制
sns.boxplot(x='性别', y='身高',data=BSdata)
# 分组箱线图，分子因子是smoker，不同的因子用不同颜色区分, 相当于分组之后又分组
sns.boxplot(x='开设', y='支出',hue='性别',data=BSdata)

##小提琴图
sns.violinplot(x='性别', y='身高',data=BSdata)
sns.violinplot(x='开设', y='支出',hue='性别',data=BSdata)

##点图
sns.stripplot(x='性别', y='身高',data=BSdata)
sns.stripplot(x='性别', y='身高',data=BSdata,jitter=True)
sns.stripplot(y='性别', x='身高',data=BSdata,jitter=True)

#计数图
sns.countplot(x='性别',data=BSdata)
sns.countplot(y='开设',data=BSdata)
sns.countplot(x='性别',hue='开设',data=BSdata)
##类似条形图 sns直接给出频数 plt还需要先计算再绘图


#分组关系图
sns.catplot(x='性别',col='开设', col_wrap=3,data=BSdata, kind='count',height=2.5, aspect=.8)
##可以使图形对齐 应用到word里面


#概率分布图
sns.distplot(BSdata['身高'], kde=True, bins=20, rug=True);
##直方图&概率密度曲线
sns.jointplot(x='身高', y='体重', data=BSdata)
sns.jointplot(BSdata['身高'],BSdata['体重'])
# 散点图&边缘直方图
sns.pairplot(BSdata[['身高','体重','支出']])
##分别对应的什么图形？？

##ggplot
from ggplot import *

##直方图
qplot('身高',data=BSdata, geom='histogram')
##条形图
qplot('开设',data=BSdata, geom='bar')

##散点图
qplot('身高','体重',data=BSdata,color='性别')

GP=ggplot(aes(x='身高',y='体重'),data=BSdata)
GP+geom_point()+geom_line()
##点+线
ggplot(BSdata,aes(x='支出'))+geom_line(aes(y='身高'))
GP+geom_point()+geom_line(x=np.random.randint(155,180,52),y=np.linspace(50,85,52))##error


GP+geom_point()
GP+geom_point()+facet_wrap('性别')
GP+geom_point()+facet_wrap('性别',nrow=1,ncol=2)
GP+geom_point()+facet_wrap('性别',nrow=1,ncol=2)+theme_bw()
##theme_gray()/xkcd()主题
ggplot(BSdata,aes(x='身高',y='体重',color='性别'))+geom_point()+theme_bw()


import pandas as pd 
import numpy as np 
import math
import matplotlib.pyplot as plt   
BSdata=pd.read_csv('./data/BSdata.csv')
plt.rcParams['font.sans-serif']=['KaiTi']

x=np.linspace(0,2*math.pi);x
plt.plot(x,np.sin(x))
plt.plot(x,np.cos(x))
plt.plot(x,np.log(x))
plt.plot(x,np.exp(x))

x=np.linspace(-math.pi,math.pi);x
plt.plot(x,np.tan(x))

t=np.linspace(0,2*math.pi)
x=2*np.sin(t);y=3*np.cos(t)
plt.plot(x,y)
plt.text(0,0,r'$\frac{x^2}{2}+\frac{y^2}{3}=1$',fontsize=20)##r??

plt.scatter(BSdata.身高,BSdata.体重,s=BSdata.支出)

from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
x=np.arange(-4,4,0.5)
y=np.arange(-4,4,0.5)
x,y=np.meshgrid(x,y)
z=np.sqrt(x**2+y**2)
ax.plot_surface(x,y,z)

from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(BSdata.身高,BSdata.体重,BSdata.支出)

##seaborn
import seaborn as sns
sns.boxplot(x=BSdata.身高)
sns.boxplot(y=BSdata.身高)
sns.boxplot(x=BSdata.性别,y=BSdata.身高)
sns.violinplot(x='开设',y='支出',hue='性别',data=BSdata)
sns.stripplot(x='性别',y='身高',data=BSdata,jitter=True)
sns.barplot(x='性别',y='身高',data=BSdata,ci=0,palette='Blues_d')
sns.countplot(x='性别',hue='开设',data=BSdata)
sns.factorplot(x='性别',col='开设',col_wrap=3,data=BSdata,kind='count',size=2.5,aspect=0.8)

sns.distplot(BSdata['身高'],kde=True,bins=20,rug=True)
sns.jointplot(x='身高',y='体重',data=BSdata)##双变量
sns.pairplot(BSdata[['身高','体重','支出']])##多变量用pairplot 对角线直方图 其他散点图

from ggplot import *
qplot('身高',data=BSdata,geom='histogram')
qplot('开设',data=BSdata,geom='bar')
qplot('身高','体重',data=BSdata,color='性别')

GP=ggplot(aes(x='身高',y='体重'),data=BSdata);GP
GP+geom_point()+geom_line()

ggplot(BSdata,aes(x='身高'))+geom_histogram()
#同 qplot('身高',data=BSdata,geom='histogram')
ggplot(BSdata,aes(x='身高',y='体重'))+geom_point()
ggplot(BSdata,aes(x='身高',y='体重',color='性别'))+geom_point()
ggplot(BSdata,aes(x='支出'))+geom_line(aes(y='身高'))
ggplot(BSdata,aes(x='支出'))+geom_line(aes(y='身高'))+geom_line(aes(y='体重'))
ggplot(BSdata,aes(x='支出'))+geom_line(aes(y='身高'))
ggplot(BSdata,aes(x='身高',y='体重'))+geom_point()+facet_wrap('性别')
ggplot(BSdata,aes(x='身高',y='体重',color='性别'))+geom_point()+theme_gray()