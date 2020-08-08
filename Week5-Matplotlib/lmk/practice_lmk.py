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

# matplotlib基础
## create a new figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig=plt.figure()
fig.suptitle('No axes on this figure')
fig,ax_lst=plt.subplots(2,2)

a=pd.DataFrame(np.random.rand(4,5),columns=list('abcde'))
a_asarray=a.values
b=np.matrix([[1,2],[3,4]])
b_asarray=np.asarray(b)

## pyplot
x=np.linspace(0,2,100)
plt.plot(x,x,label='linear')
plt.plot(x,x*2,label='quadratic')
plt.plot(x,x*3,label='cubic')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title('Simple Plot')
plt.legend()
plt.show()

## backend
import matplotlib
matplotlib.use('pdf')

plt.ioff() # non-interactive
plt.plot([1.6,2.7])
plt.show()

## pyplot tutorial
plt.plot([1,2,3,4],[1,4,9,16],'ro')
plt.axis([0,6,0,20])
plt.show()

t=np.arange(0.,5.,0.2)
plt.plot(t,t,'r--',t,t**2,'bs',t,t**3,'g^') # red dashes,blue squares and green triangles
plt.show()

data={'a':np.arange(50),
      'c':np.random.randint(0,50,50),
      'd':np.random.randn(50)}
data['b']=data['a']+10*np.random.randn(50)
data['d']=np.abs(data['d'])*100

plt.scatter('a','b',c='c',s='d',data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()

names=['group_a','group_b','group_c']
values=[1,10,100]
plt.figure(figsize=(9,3))
plt.subplot(131)
plt.bar(names,values)
plt.subplot(132)
plt.scatter(names,values)
plt.subplot(133)
plt.plot(names,values)
plt.suptitle('Categorical Plotting')
plt.show()

### line properties
line=plt.plot(x,y,'-')
line.set_antialiased(False) # turn off antialiasing

lines=plt.plot(x1,y1,x2,y2)
plt.setp(lines,color='r',linewidth=2.0) # use keywords args
plt.setp(lines,'color','r','linewidth',2.0) # or MATLAB style string value pairs

lines=plt.plot([1,2,3])
plt.setp(lines)

### multiple figures and axes
def f(t):
      return np.exp(-t)*np.cos(2*np.pi*t)
t1=np.arange(0.0,5.0,0.1)
t2=np.arange(0.0,5.0,0.02)
plt.figure()
plt.subplot(211)
plt.plot(t1,f(t1),'bo',t2,f(t2),'k')
plt.subplot(212)
plt.plot(t2,np.cos(2*np.pi*t2),'r--')
plt.show()

plt.figure(1) # the first figure
plt.subplot(211)
plt.plot([1,2,3])
plt.subplot(212)
plt.plot([4,5,6])
plt.figure(2) # a second figure
plt.plot([4,5,6])
plt.figure(1)
plt.subplot(211)
plt.title('Easy as 1,2,3')

### working with text
t=plt.xlabel('my data',fontsize=14,color='red')

mu,sigma=100,15
x=mu+sigma*np.random.randn(10000)
n,bins,patches=plt.hist(x,50,density=1,facecolor='g',alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60,.025,r'$\mu=100,\ \sigma=15$')
plt.axis([40,160,0,0.03])
plt.grid(True)
plt.show()

ax=plt.subplot(111)
t=np.arange(0.0,5.0,0.01)
s=np.cos(2*np.pi*t)
line=plt.plot(t,s,lw=2)
plt.annotate('local max',xy=(2,1),xytext=(3,1.5),arrowprops=dict(facecolor='black',shrink=0.05)) # xy为需要注释的位置，xytext为注释文本的位置
plt.ylim(-2,2)
plt.show()

## object-oriented API
from matplotlib.ticker import FuncFormatter
data= {'Barton LLC': 109438.50,
       'Frami, Hills and Schmidt': 103569.59,
       'Fritsch, Russel and Anderson': 112214.71,
       'Jerde-Hilpert': 112591.43,
       'Keeling LLC': 100934.30,
       'Koepp Ltd': 103660.54,
       'Kulas Inc': 137351.96,
       'Trantow-Barrows': 123381.38,
       'White-Trantow': 135841.99,
       'Will LLC': 104437.60}
group_data=list(data.values())
group_names=list(data.keys())
group_mean=np.mean(group_data)

fig,ax=plt.subplots()
ax.barh(group_names,group_data)

print(plt.style.available)
plt.style.use('ggplot')

### customizing the plot
fig,ax=plt.subplots()
ax.barh(group_names,group_data)
labels=ax.get_xticklabels()

fig,ax=plt.subplots()
ax.barh(group_names,group_data)
labels=ax.get_xticklabels()
plt.setp(labels,rotation=45,horizontalalignment='right') # 标签右边旋转45度

plt.rcParams.update({'figure.autolayout':True})
fig,ax=plt.subplots()
ax.barh(group_names,group_data)
labels=ax.get_xticklabels()
plt.setp(labels,rotation=45,horizontalalignment='right')

def currency(x,pos):
    """The two args are the value and tick position"""
    if x>=1e6:
        s = '${:1.1f}M'.format(x*1e-6)
    else:
        s='${:1.0f}K'.format(x*1e-3)
    return s
formatter=FuncFormatter(currency)
fig,ax=plt.subplots(figsize=(6,8))
ax.barh(group_names,group_data)
labels=ax.get_xticklabels()
plt.setp(labels,rotation=45,horizontalalignment='right')
ax.set(xlim=[-10000,140000],xlabel='Total Revenue',ylabel='Company',title='Company Revenue')
ax.xaxis.set_major_formatter(formatter)

### combining multiple visualizations
fig,ax=plt.subplots(figsize=(8,8))
ax.barh(group_names,group_data)
labels=ax.get_xticklabels()
plt.setp(labels,rotation=45,horizontalalignment='right')
ax.axvline(group_mean,ls='--',color='r')
for group in [3,5,8]:
      ax.text(145000,group,'New Company',fontsize=10,verticalalignment='center')
ax.title.set(y=1.05)
ax.set(xlim=[-10000,140000],xlabel='Total Revenue',ylabel='Company',title='Company Revenue')
ax.xaxis.set_major_formatter(formatter)
ax.set_xticks([0,25e3,50e3,75e3,100e3,125e3])
fig.subplots_adjust(right=.1)
plt.show() ### ？？？

### saving plots
print(fig.canvas.get_supported_filetypes())
fig.savefig('sales.png',transparent=False,dpi=80,bbox_iches='tight') # transparent=True表示保存图片的背景为透明，dpi控制图像的分辨率，bbox_iches='tight'表示将图形边界与我们的图相匹配

## style sheets
import matplotlib as mpl
plt.style.use('ggplot')
data=np.random.randn(50)
print(plt.style.available)
# plt.style.use(['dark_background','presentation'])

with plt.style.context('dark_background'):
    plt.plot(np.sin(np.linspace(0,2*np.pi)),'r-o')
plt.show()

## matplotlib rcParams
mpl.rcParams['lines.linewidth']=2
mpl.rcParams['lines.color']='r'
plt.plot(data)

mpl.rc('lines',linewidth=4,color='g')
plt.plot(data)

import matplotlib
matplotlib.matplotlib_fname()

## artist tutorial
fig=plt.figure()
ax=fig.add_subplot(2,1,1) # two rows,one column,first plot

fig2=plt.figure()
ax2=fig2.add_axes([0.15,0.1,0.7,0.3])
t=np.arange(0.0,1.0,0.01)
s=np.sin(2*np.pi*t)
line=ax.plot(t,s,color='blue',lw=2)
type(ax.lines)
len(ax.lines)
type(line)

### customizing objects
plt.getp(fig) # 查看属性
plt.getp(ax)

a=line.get_alpha() # 'list' object has no attribute 'get_alpha'
line.set_alpha(0.5*a)
line.set(alpha=0.5,zorder=2)

## object containers
### figure containers
fig=plt.figure()
ax1=fig.add_subplot(211)
ax2=fig.add_axes([0.1,0.1,0.7,0.3])
print(fig.axes)

for ax in fig.axes:
    ax.grid(True)

### axis containers
fig,ax=plt.subplots()
axis=ax.xaxis
axis.get_ticklocs()

axis.get_ticklabels()
axis.get_ticklines()
axis.get_ticklines(minor=True)

### tick containers
np.random.seed(19680801)
fig,ax=plt.subplots()
ax.plot(100*np.random.rand(20))
formatter=ticker.FormatStrFormatter('$%1.2f') # ticker
ax.yaxis.set_major_formatter(formatter)

for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_visible(False)
    tick.label2.set_visible(True)
    tick.label2.set_color('green')
plt.show()

## legend
line=ax.plot([1,2,3],label='Inline label')
ax.legend()
line,=ax.plot([1,2,3])
line.set_label('Label via method')
ax.legend()

ax.plot([1,2,3])
ax.legend(['A simple line'])

a=b=np.arange(0,3,.02)
c=np.exp(a)
d=c[::-1]
fig,ax=plt.subplots()
ax.plot(a,c,'k--',label='Model length')
ax.plot(a,d,'k:',label='Data length')
ax.plot(a,c+d,'k',label='Total message length')
legend=ax.legend(loc='upper center',shadow='True',fontsize='x-large')
legend.get_frame().set_facecolor('C0')
plt.show()

## tight layout and constrained layout
### tight layout
plt.rcParams['savefig.facecolor']='0.8'
def example_plot(ax,fontsize=12):
    ax.plot([1,2])

    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label',fontsize=fontsize)
    ax.set_ylabel('y-label',fontsize=fontsize)
    ax.set_title('Title',fontsize=fontsize)

plt.close('all')
fig,ax=plt.subplots()
example_plot(ax,fontsize=24)

fig,ax=plt.subplots()
example_plot(ax,fontsize=24)
plt.tight_layout() # 调整布局

### mutiple subplots
plt.close('all')
fig=plt.figure()
ax1=plt.subplot(221)
ax2=plt.subplot(223)
ax3=plt.subplot(122)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
plt.tight_layout()

### constrained layout
fig,ax=plt.subplots(constrained_layout=False)
example_plot(ax,fontsize=24)

fig,ax=plt.subplots(constrained_layout=True)
example_plot(ax,fontsize=24)

fig,axs=plt.subplots(2,2,constrained_layout=False)
for ax in axs.flat:
    example_plot(ax)

fig,axs=plt.subplots(2,2,constrained_layout=True) # 调整多图布局使得标签不会重叠
for ax in axs.flat:
    example_plot(ax)

## text in matplotlib plots
fig=plt.figure()
ax=fig.add_subplot(111)
fig.subplots_adjust(top=0.85) # 调整图形高度
fig.suptitle('bold figure suptitle',fontsize=14,fontweight='bold') # 添加图标题
ax.set_title('axes title') # 添加轴标题
ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')
ax.axis([0,10,0,10]) # 修改坐标轴范围
ax.text(3,8,'boxed italics text in data coords',style='italic',bbox={'facecolor':'red','alpha':0.5,'pad':10}) # 在指定位置添加文本，alpha控制透明度，pad控制内边距
ax.text(2,6,r'an equation: $E=mc^2$',fontsize=15)
ax.text(0.95,0.01,'colored text in axes coords',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green',fontsize=15)

ax.plot([2],[1],'o')
ax.annotate('annotate',xy=(2,1),xytext=(3,4),arrowprops=dict(facecolor='black',shrink=0.05)) # shrink控制箭头离注释点和注释文字的距离
plt.show()

plt.getp(ax.texts)

## annotations
### basic annotation
ax=plt.subplot(111)
t=np.arange(0.0,5.0,0.01)
s=np.cos(2*np.pi*t)
line=plt.plot(t,s,lw=2)
ax.annotate('local max',xy=(3,1),xycoords='data',
            xytext=(0.8,0.95),textcoords='axes fraction',
            arrowprops=dict(facecolor='black',shrink=0.05),
            horizontalalignment='right',verticalalignment='top')
plt.ylim(-2,2)
plt.show()

### advanced annotation
#### annotating with text with box
bbox_props=dict(boxstyle='rarrow,pad=0.3',fc='cyan',ec='b',lw=2)
t=ax.text(0.5,0.5,'Direction',ha='center',va='center',rotation=45,size=15,bbox=bbox_props)
bb=t.get_bbox_patch()
bb.set_boxstyle('rarrow',pad=0.6)

#### fancybox list
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatch
from matplotlib.patches import FancyBboxPatch

styles=mpatch.BoxStyle.get_styles()
spacing=1.2
figheight=(spacing*len(styles)+.5)
fig=plt.figure(figsize=(4/1.5,figheight/1.5))
fontsize=0.3*72

for i,stylename in enumerate(sorted(styles)):
    fig.text(0.5,(spacing*(len(styles)-i)-0.5)/figheight,
             stylename,ha='center',size=fontsize,
             transform=fig.transFigure,
             bbox=dict(boxstyle=stylename,fc='w',ec='k'))
plt.show()

## the mplot3d toolkit
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

plt.rcParams['legend.fontsize']=10
fig=plt.figure()
ax=fig.gca(projection='3d')
theta=np.linspace(-4*np.pi,4*np.pi,100)
z=np.linspace(-2,2,100)
r=z**2+1
x=r*np.sin(theta)
y=r*np.cos(theta)
ax.plot(x,y,z,label='parametric curve')
ax.legend()
plt.show()

### 支持的3D图形类型
线形图：Axes3D.plot(self, xs, ys, *args, zdir='z', **kwargs)
散点图：Axes3D.scatter(self, xs, ys, zs=0, zdir='z', s=20, c=None, depthshade=True, *args, **kwargs)
线框图：Axes3D.plot_wireframe(self, X, Y, Z, *args, **kwargs)
曲面图：Axes3D.plot_surface(self, X, Y, Z, *args, norm=None, vmin=None, vmax=None, lightsource=None, **kwargs)
三维曲面图：Axes3D.plot_trisurf(self, *args, color=None, norm=None, vmin=None, vmax=None, lightsource=None, **kwargs)
等高线图：Axes3D.contour(self, X, Y, Z, *args, extend3d=False, stride=5, zdir='z', offset=None, **kwargs)
填充等高线图：Axes3D.contourf(self, X, Y, Z, *args, zdir='z', offset=None, **kwargs)
多边形图：Axes3D.add_collection3d(self, col, zs=0, zdir='z')
条形图：Axes3D.bar(self, left, height, zs=0, zdir='z', *args, **kwargs)
绘制箭头：Axes3D.quiver(X, Y, Z, U, V, W, /, length=1, arrow_length_ratio=.3, pivot='tail', normalize=False, **kwargs)
添加文本：Axes3D.text(self, x, y, z, s, zdir=None, **kwargs)