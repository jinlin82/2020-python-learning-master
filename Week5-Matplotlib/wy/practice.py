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

# matplotlib.pdf #

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

##图形
fig=plt.figure() # an empty figure with no axes
fig.suptitle('No axes on this figure') # add a title so we know which it is
fig,ax_lst=plt.subplots(2,2) # a figure with a 2x2 grid of axes

##绘图函数的输入类型
a=pd.DataFrame(np.random.rand(4,5),columns=list('abcde'))
a
a_asarray=a.values

b=np.matrix([[1,2],[3,4]])
b
b_asarray=np.asarray(b)

## pyplot
x=np.linspace(0,2,100)
x
plt.plot(x,x,label='linear')
plt.plot(x,x**2,label='quadratic')
plt.plot(x,x**3,label='cubic')

plt.xlabel('x label')
plt.ylabel('y label')
plt.title('simple plot')
plt.legend()

plt.show()

## backends
import matplotlib
matplotlib.use('pdf') ###generate postscript output by default

## what is interactive mode?

#matplotlib.pyplot.ion() turn on interactive mode
#matplotlib.pyplot.ioff() turn off interactive mode

plt.ioff()
plt.plot([1.6,2.7])

plt.show()

## formatting the style of your plot
plt.plot([1,2,3,4],[1,4,9,16],'ro')
plt.axis([0,6,0,20])
plt.show()

## use numpy arrays
t=np.arange(0,5,0.2) #evenly sampled time at 200ms intervals
t
plt.plot(t,t,'r--',t,t**2,'bs',t,t**3,'g^')
plt.show()

## plotting with keyword strings
data={'a':np.arange(50),'c':np.random.randint(0,50,50),'d':np.random.randn(50)}
data['b']=data['a']+10*np.random.randn(50)
data['d']=np.abs(data['d'])*100

plt.scatter('a','b',c='c',s='d',data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.legend()
plt.show()

## plotting with categorical variables
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

## controlling line properties
x=[0,1,2,3];x
y=[0,1,4,9];y
line,=plt.plot(x,y,'-')  #逗号？
line.set_antialiased(False)

x1=[0,1,2,3]
y1=[0,1,4,9]
x2=[3,4,5,6,7]
y2=[9,16,25,36,49]
lines=plt.plot(x1,y1,x2,y2)
plt.setp(lines,color='r',linewidth=2.0) #use keyword args
lines=plt.plot(x1,y1,x2,y2)
plt.setp(lines,'color','r','linewidth',2.0) # or MATLAB style string value pairs

lines=plt.plot([1,2,3])
plt.setp(lines)

## working with multiple figures and axes
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1=np.arange(0,5,0.1)
t2=np.arange(0,5,0.02)
plt.figure()
plt.subplot(211)
plt.plot(t1,f(t1),'bo',t2,f(t2),'k')
plt.subplot(212)
plt.plot(t2,np.cos(2*np.pi*t2),'r--')
plt.show()

plt.figure(1) # the first figure
plt.subplot(211) # the first subplot in the first figure
plt.plot([1,2,3])
plt.subplot(212) # the second subplot in the first figure
plt.plot([4,5,6])

plt.figure(2) # the second figure
plt.plot([4,5,6]) # creates a subplot(111) by default

plt.figure(1) # figure 1 current; subplot(212) still current
plt.subplot(211) #make subplot(211) in figure1 current 
plt.title('Easy as 1,2,3') # subplot 211 title

## working with text
t=plt.xlabel('my data',fontsize=14,color='red')

mu,sigma=100,15
x=mu+sigma*np.random.randn(10000)

n,bins,patches=plt.hist(x,50,density=1,facecolor='g',alpha=0.75)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60,0.025,r'$\mu=100,\ \sigma=15$')
plt.axis([40,160,0,0.03])
plt.grid(True)
plt.show()

## annotating text
ax=plt.subplot(111)

t=np.arange(0,5,0.01)
s=np.cos(2*np.pi*t)
line,=plt.plot(t,s,lw=2) ##逗号

plt.annotate('local max',xy=(2,1),xytext=(3,1.5),arrowprops=dict(facecolor='black',shrink=0.05),)

plt.ylim(-2,2)
plt.show()

## object-orient API
from matplotlib.ticker import FuncFormatter

data={'Barton LLC':109438.50,'Frami, Hills and Schmidt':103569.59, 'Fritsch, Russel and Anderson':112214.71,'Jerde-Hilpert':112591.43,'Keeling LLC':100934.30,'Koepp Ltd':103660.54,'Kulas Inc':137351.96,'Trantow-Barrows':123381.38,'White-Trantow':135841.99,'Will LLC':104437.60}

group_data=list(data.values())
group_names=list(data.keys())
group_mean=np.mean(group_data)

fig,ax=plt.subplots()
ax.barh(group_names,group_data)

## controlling the style
print(plt.style.available) # to see a list of styles
plt.style.use('ggplot') #activate a style

##customizing the plot
fig,ax=plt.subplots()
ax.barh(group_names,group_data)
labels=ax.get_xticklabels()
plt.setp(labels,rotation=45,horizontalalignment='right')

plt.rcParams.update({'figure.autolayout':True})
fig,ax=plt.subplots()
ax.barh(group_names,group_data)
labels=ax.get_xticklabels()
plt.setp(labels,rotation=45,horizontalalignment='right')

def currency(x,pos):
    """The two args are the value and tick position"""
    if x>=1e6:
        s='${:1.1f}M'.format(x*1e-6)
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

## combining multiple visualizations

fig,ax=plt.subplots(figsize=(8,8))
ax.barh(group_names,group_data)
labels=ax.get_xticklabels()
plt.setp(labels,rotation=45,horizontalalignment='right')

ax.axvline(group_mean,ls='--',color='r') #add a vertical line, here we set the style in the function call

for group in [3,5,8]:
    ax.text(145000,group,'New Company',fontsize=10,verticalalignment='center') #annotate new companies
    
ax.title.set(y=1.05)
ax.set(xlim=[-10000,140000],xlabel='Total Revenu',ylabel='Company',title='Company Revenue')
ax.xaxis.set_major_formatter(formatter)
ax.set_xticks([0,25e3,50e3,75e3,100e3,125e3])
fig.subplots_adjust(right=1)  #######
plt.show()

## saving plots
print(fig.canvas.get_supported_filetypes())

fig.savefig('sales.png',transparent=False,dpi=80,bbox_inches='tight')

## using style sheet
import matplotlib as mpl
plt.style.use('ggplot')
data=np.random.randn(50)

## composing style
#plt.style.use(['dark_background','presentation'])

## temporary styling
with plt.style.context('dark_background'):
    plt.plot(np.sin(np.linspace(0,2*np.pi)),'r-o')
plt.show()

## dynamic rc settings
mpl.rcParams['lines.linewidth']=2
mpl.rcParams['lines.color']='r'
plt.plot(data)

mpl.rc('lines',linewidth=4,color='g') ####
plt.plot(data)

## the matplotlibrc file
import matplotlib
matplotlib.matplotlib_fname()

## Artist简介

## how to create Figure instance
fig=plt.figure()
ax=fig.add_subplot(211) #two rows, one column, first plot

##axes
fig2=plt.figure()
ax2=fig2.add_axes([0.15,0.1,0.7,0.3])

fig=plt.figure()
ax=fig.add_subplot(211)
t=np.arange(0,1,0.01)
s=np.sin(2*np.pi*t)
line,=ax.plot(t,s,color='blue',lw=2)

type(ax.lines)
len(ax.lines)
type(line)

## get Properties list
plt.getp(fig)
plt.getp(ax)

## get and set properties
a=line.get_alpha()
line.set_alpha(0.5*a)
line.set(alpha=0.5,zorder=2)  ######

## Figure container
fig=plt.figure()
ax1=fig.add_subplot(211)
ax2=fig.add_axes([0.1,0.1,0.7,0.3])
print(fig.axes)

for ax in fig.axes:
    ax.grid(True)

## Axis container
fig,ax=plt.subplots()
axis=ax.xaxis
axis.get_ticklocs()
fig,ax=plt.subplots()
axis=ax.xaxis
axis.get_ticklabels()
fig,ax=plt.subplots()
axis=ax.xaxis
axis.get_ticklines()
fig,ax=plt.subplots()
axis=ax.xaxis
axis.get_ticklines(minor=True)

## Tick container
np.random.seed(19680801)

fig,ax=plt.subplots()
ax.plot(100*np.random.rand(20))
formatter=matplotlib.ticker.FormatStrFormatter('$%1.2f')
ax.yaxis.set_major_formatter(formatter)

for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_visible(False)
    tick.label2.set_visible(True)
    tick.label2.set_color('green')

plt.show()

## automatic detection of elements to be shown in the legend
fig,ax=plt.subplots()
line,=ax.plot([1,2,3],label='Inline label')
ax.legend()

fig,ax=plt.subplots()
line,=ax.plot([1,2,3])
line.set_label('label via method')
ax.legend()

## labeling existing plot elements
plt.plot([1,2,3])
plt.legend(['A simple line'])

## explicitly defining the elements in the legend
a=b=np.arange(0,3,0.2)
c=np.exp(a)
d=c[::-1]

fig,ax=plt.subplots()
ax.plot(a,c,'k--',label='Model length')
ax.plot(a,d,'k:',label='Data length')
ax.plot(a,c+d,'k',label='Total message length')
legend = ax.legend(loc='upper center',shadow=True,fontsize='x-large') # create plots with pre-defined labels

legend.get_frame().set_facecolor('C0')
plt.show()

## tight layout
plt.rcParams['savefig.facecolor']='0.8'

def example_plot(ax,fontsize=12):
    ax.plot([1,2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label',fontsize=fontsize)
    ax.set_ylabel('y-label',fontsize=fontsize)
    ax.set_title('title',fontsize=fontsize)

plt.close('all')
fig,ax=plt.subplots()
example_plot(ax,fontsize=24)

fig,ax=plt.subplots()
example_plot(ax,fontsize=24)
plt.tight_layout()

plt.close('all')

fig=plt.figure()
ax1=plt.subplot(221)
ax2=plt.subplot(223)
ax3=plt.subplot(122)

example_plot(ax1)
example_plot(ax2)
example_plot(ax3)

plt.tight_layout()

## constrained layout
fig,ax=plt.subplots(constrained_layout=False)
example_plot(ax,fontsize=24)

fig,ax=plt.subplots(constrained_layout=True)
example_plot(ax,fontsize=24)

fig,axs=plt.subplots(2,2,constrained_layout=False)
for ax in axs.flat:
    example_plot(ax)

fig,axs=plt.subplots(2,2,constrained_layout=True)
for ax in axs.flat:
    example_plot(ax)

## text in matplotlib plots
fig=plt.figure()
ax=fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

fig.suptitle('bold figure suptitle',fontsize=14,fontweight='bold')
ax.set_title('axes title')
ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel') #set titles for the figure and the subplot respectively

ax.axis([0,10,0,10]) #set both x- and y-axis limits to [0,10] instead of default [0,1]

ax.text(3,8,'boxed italics text in data coords',style='italic',bbox={'facecolor':'red','alpha':0.5,'pad':10})
ax.text(2,6,r'an equation: $E=mc^2$',fontsize=15)
ax.text(0.95,0.01,'colored text in axes coords',verticalalignment='bottom',horizontalalignment='right',transform=ax.transAxes,color='green',fontsize=15)
ax.plot([2],[1],'o')
ax.annotate('annotate',xy=(2,1),xytext=(3,4),arrowprops=dict(facecolor='black',shrink=0.05))
plt.show()
plt.getp(ax.texts)

## basic annotation
ax=plt.subplot(111)

t=np.arange(0,5,0.01)
s=np.cos(2*np.pi*t)
line,=plt.plot(t,s,lw=2)

ax.annotate('local max',xy=(3,1),xycoords='data',xytext=(0.8,0.95),textcoords='axes fraction',arrowprops=dict(facecolor='black',shrink=0.05),horizontalalignment='right',verticalalignment='top')

plt.ylim(-2,2)
plt.show()

## advanced annotation
bbox_props=dict(boxstyle="rarrow,pad=0.3",fc='cyan',ec='b',lw=2)
t=ax.text(0.5,0.5,'Direction',ha='center',va='center',rotation=45,size=15,bbox=bbox_props)
bb=t.get_bbox_patch()
bb.set_boxstyle('rarrow',pad=0.6) ######

## fancybox list
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatch
from matplotlib.patches import FancyBboxPatch

styles=mpatch.BoxStyle.get_styles()
spacing=1.2
figheight=(spacing*len(styles)+0.5)
fig=plt.figure(figsize=(4/1.5,figheight/1.5))
fontsize=0.3*72

for i, stylename in enumerate(sorted(styles)):
    fig.text(0.5,(spacing*(len(styles)-i)-0.5)/figheight,stylename,ha='center',size=fontsize,transform=fig.transFigure,bbox=dict(boxstyle=stylename,fc='w',ec='k'))
plt.show() ######无图

##Axes3D object
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d') #三维坐标

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

