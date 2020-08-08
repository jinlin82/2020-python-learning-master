import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
BSdata = pd.read_csv('./BSdata.csv')
BSdata.describe()

BSdata[['性别','开设','课程','软件']].describe()

T1 = BSdata.性别.value_counts();T1
T1/sum(T1)*100
#BSdata.crosstab()
BSdata.性别.value_counts(normalize=True)*100

BSdata.身高.mean()
BSdata.身高.median()
BSdata.身高.max()-BSdata.身高.min()
BSdata.身高.var()
BSdata.身高.std()
BSdata.身高.quantile(0.75)-BSdata.身高.quantile(0.25)
BSdata.身高.skew()
BSdata.身高.kurt()

def stats(x):
    stats = [x.count(),x.min(),x.quantile(0.25),x.mean(),x.median(),x.quantile(0.75),x.max(),x.max()-x.min(),x.var(),x.std(),x.skew(),x.kurt()]
    stats = pd.Series(stats,index=['Count','min','Q1(25%)','Mean','Median','Q3(75%)','Max','Range','Var','Std','Skew','Kurt'])
    return(stats)
stats(BSdata.身高)
stats(BSdata.支出)

import matplotlib.pyplot as plt 
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(6,5))
plt.show()

X = ['A','B','C','D','E','F','G']
Y = [1,4,7,3,2,5,6]
plt.bar(X,Y) #条图
plt.show()
plt.savefig('abc',format='pdf')

plt.pie(Y,labels=X)#饼图
plt.show()

plt.plot(X,Y)
plt.show()

#直方图hist
plt.hist(BSdata.身高)
plt.xlabel('bbb')
plt.show()

import matplotlib.pyplot as plt
plt.hist([1,2,3,4])
plt.xlabel('abc')

plt.hist(BSdata.身高,density=True)
plt.show()

#scatter散点图
plt.scatter(BSdata.身高,BSdata.体重)
plt.show()

plt.ylim(0,8)
plt.xlabel('names');plt.ylabel('values')
plt.xticks(range(len(X)),X)
plt.show()

plt.plot(X,Y,linestyle='--',marker='o')
plt.show()

plt.plot(X,Y,'o--')
plt.axvline(x=1)
plt.axhline(y=4)
plt.show()

plt.plot(X,Y)
plt.text(2,7,'peakpoint')
plt.show()

plt.plot(X,Y,label=u'折线')

s = [0.1,0.4,0.7,0.3,0.2,0.5,0.6]
plt.bar(X,Y,yerr=s,error_kw={'capsize':5})
plt.show()

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.bar(X,Y)
plt.subplot(1,2,2)
plt.plot(Y)
plt.show()

fig,ax = plt.subplots(1,2,figsize=(14,6))
ax[0].bar(X,Y)
ax[1].plot(X,Y)
plt.show()

fig,ax = plt.subplots(2,2,figsize=(5,10))
ax[0,0].bar(X,Y)
ax[0,1].pie(Y,labels=X)
ax[1,0].plot(Y)
ax[1,1].plot(Y,'-',linewidth=3)
plt.show()

plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
BSdata['体重'].plot(kind='line')
plt.subplot(2,2,2)
BSdata['体重'].plot(kind='hist')
plt.subplot(2,2,3)
BSdata['体重'].plot(kind='box')
plt.subplot(2,2,4)
BSdata['体重'].plot(kind='density',title='Density')
plt.show()

#定性数据
T1 = BSdata['开设'].value_counts();T1
pd.DataFrame({'频数':T1,'频率':T1.T1.sum()*100})
T1.plot(kind='bar')
T1.plot(kind='pie')

f = BSdata['开设'].value_counts()
sum(f)
BSdata['开设'].value_counts(normalize=True)


def tab(x,plot=False):
    f = x.value_counts();f
    s = sum(f)
    p = round(f/s*100,3);p
    T1 = pd.concat([f,p],axis=1)
    T1.columns = ['例数','构成比']
    T2 = pd.DataFrame({'例数':s,'构成比':100.00},index=['合计'])
    Tab = T1.append(T2)
    if plot:
        fig,ax = plt.subplots(2,1,figsize=(8,15))
        ax[0].bar(f.index,f)
        ax[1].pie(p,labels=p.index,autopct='%1.2f%%')
    return(round(Tab,3))

def tab(x,plot=False): #计数频数表
    f=x.value_counts();f
    s=sum(f)
    p=round(f/s*100,3);p
    T1=pd.concat([f,p],axis=1)
    T1.columns=['例数','构成比']
    T2=pd.DataFrame({'例数':s,'构成比':100.00},index=['合计'])
    Tab=T1.append(T2)
    if plot:
        fig,ax = plt.subplots(2,1,figsize=(8,15))
        ax[0].bar(f.index,f); # 条图
        ax[1].pie(p,labels=p.index,autopct='%1.2f%%');
    return(round(Tab,3))

tab(BSdata.开设,True)  #Q 报错 显示函数未定义？？？

pd.cut(BSdata.身高,bins=10).value_counts()
pd.cut(BSdata.身高,bins=10).value_counts().plot(kind='bar')
plt.show()

pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts()

pd.cut(BSdata.支出,bins=[0,10,30,100]).value_counts().plot(kind='bar')
plt.show()

def freq(X,bins=10):
    H = plt.hist(X,bins);
    a = H[1][:-1];a
    b = H[1][1:];b
    f = H[0];f
    p = f/sum(f)*100;p
    cp = np.cumsum(p);cp
    Freq = pd.DataFrame([a,b,f,p,cp])
    Freq.index = ['下限','上限','频数','频率','累计频数（%）']
    return(round(Freq.T,2))

freq(BSdata.体重)
X = BSdata.体重

pd.crosstab(BSdata.开设,BSdata.课程)
pd.crosstab(BSdata.开设,BSdata.课程,margins=True)

pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='index')
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='columns')
pd.crosstab(BSdata.开设,BSdata.课程,margins=True,normalize='all').round(3)

T2 = pd.crosstab(BSdata.开设,BSdata.课程);T2
T2.plot(kind='bar')
T2.plot(kind='bar',stacked=True)

BSdata.groupby(['性别'])
type(BSdata.groupby(['性别']))
BSdata.groupby(['性别'])['身高'].mean()
BSdata.groupby(['性别'])['身高'].size()
BSdata.groupby(['性别','开设'])['身高'].mean()

BSdata.groupby(['性别'])['身高'].agg([np.mean,np.std])

BSdata.groupby(['性别'])['身高','体重'].apply(np.mean)
BSdata.groupby(['性别','开设'])['身高','体重'].apply(np.mean)

BSdata.pivot_table(index=['性别'],values=['学号'],aggfunc=len)
BSdata.pivot_table(values=['学号'],index=['性别','开设'],aggfunc=len)
BSdata.pivot_table(values=['学号'],index=['开设'],columns=['性别'],aggfunc=len)

BSdata.pivot_table(index=['性别'],values=['身高'],aggfunc=np.mean)
BSdata.pivot_table(index=['性别'],values=['身高'],aggfunc=[np.mean,np.std])
BSdata.pivot_table(index=['性别'],values=['身高','体重'])

BSdata.pivot_table('学号',['性别','开设'],'课程',aggfunc=len,margins=True,margins_name='合计')
BSdata.pivot_table(['身高','体重'],['性别','开设'],aggfunc=[len,np.mean,np.std])

BSdata.学号

#05
#初等函数图

import math 
import numpy as np 
import matplotlib.pyplot as plt
x = np.linspace(0,2*math.pi);x
fig,ax = plt.subplots(2,2,figsize=(15,12))
ax[0,0].plot(x,np.sin(x))
ax[0,1].plot(x,np.cos(x))
ax[1,0].plot(x,np.log(x))
ax[1,1].plot(x,np.exp(x))
plt.show()

#极坐标图
t = np.linspace(0,2*math.pi)
x = 2*np.sin(t)
y = 3*np.cos(t)
plt.plot(x,y)
plt.text(0,0,r'$\frac{x^2}{2}+\frac{y^2}{3}=1$',fontsize=15)#fontsize字体大小
plt.show()

#气泡图
import pandas as pd 
BSdata = pd.read_csv('./BSdata.csv',encoding='utf-8')
plt.scatter(BSdata['身高'],BSdata['体重'],s=BSdata['支出'])

#三维曲面图
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X = np.linspace(-4,4,20) 
Y = np.linspace(-4,4,20)
X,Y = np.meshgrid(X,Y)
Z = np.sqrt(X**2+Y**2)
ax.plot_surface(X,Y,Z)
plt.show()

#三维散点图
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(BSdata['身高'],BSdata['体重'],BSdata['支出'])
plt.show()

import seaborn as sns 

sns.boxplot(x=BSdata['身高'])#横着放的箱线图
sns.boxplot(y=BSdata['身高'])#竖着放的箱线图
sns.boxplot(x='性别',y='身高',data=BSdata)#x为分组因子
sns.boxplot(x='开设',y='支出',hue='性别',data=BSdata)

sns.violinplot(x='性别',y='身高',data=BSdata)
sns.violinplot(x='开设',y='支出',hue='性别',data=BSdata)

sns.stripplot(x='性别',y='身高',data=BSdata)
sns.stripplot(x='性别',y='身高',data=BSdata,jitter=True)
sns.stripplot(y='性别',x='身高',data=BSdata,jitter=True)

sns.barplot(x='性别',y='身高',data=BSdata,ci=0,palette='Blues_d')

sns.countplot(x='性别',data=BSdata)
sns.countplot(y='开设',data=BSdata)
sns.countplot(x='性别',hue='开设',data=BSdata)

sns.catplot(x='性别',col='开设',col_wrap=3,data=BSdata,kind='count',height=2.5,aspect=0.8)

sns.distplot(BSdata['身高'],kde=True,bins=20,rug=True)
sns.jointplot(x='身高',y='体重',data=BSdata)
sns.pairplot(BSdata[['身高','体重','支出']])

from ggplot import *
import matplotlib.pyplot as plt #基本绘图包
plt.rcParams['font.sans-serif']=['KaiTi'] #SimHei黑体
plt.rcParams['axes.unicode_minus']=False #正常显示图中负号

qplot('身高',data=BSdata,geom='histogram')
qplot('开设',data=BSdata,geom='bar')

#散点图
qplot('身高','体重',data=BSdata,color='性别')

GP = ggplot(aes(x='身高',y='体重'),data=BSdata);GP

#直方图
ggplot(BSdata,aes(x='身高'))+ geom_histogram()

#散点图
ggplot(BSdata,aes(x='身高',y='体重'))+ geom_point()

#线图
ggplot(BSdata,aes(x='支出'))+geom_line(aes(y='身高'))

#分面图
ggplot(BSdata,aes(x='身高',y='体重'))+ geom_point() + facet_wrap('性别')

#添加主题
ggplot(BSdata,aes(x='身高',y='体重',color='性别'))+ geom_point() + theme_bw()


###matplotlib

import matplotlib.pyplot as plt
import numpy as np 

fig = plt.figure() #an empty figure with no axes
fig.suptitle('No axes on this figure') # Add a title so we know which it is
plt.show()
##jupyter不能显示

fig,ax_lst = plt.subplots(2,2) # a figure with a 2*2 grid of Axes

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
 
a = pd.DataFrame(np.random.rand(4,5),columns=list('abcde'))
a_asarray = a.values #把DataFrame降维为Array
b = np.matrix([[1,2],[3,4]])
b_asarray = np.asarray(b)

###pyplot
x = np.linspace(0,2,100)
plt.plot(x,x,label='linear')
plt.plot(x,x**2,label='quadratic')
plt.plot(x,x**3,label='cubic')

plt.xlabel('x label')
plt.ylabel('y label')
plt.title('Simple Plot')
plt.legend()
plt.show()

import matplotlib
matplotlib.use('pdf')###generate postscirpt output by default

import matplotlib.pyplot as plt 
plt.ioff()
plt.plot([1.6,2.7])

import matplotlib.pyplot as plt 
plt.plot([1,2,3,4],[1,4,9,16],'ro')
plt.axis([0,6,0,20])#x轴y轴的范围
plt.show()

import numpy as np 
t = np.arange(0,5,0.2)
plt.plot(t,t,'r--',t,t**2,'bs',t,t**3,'g^')
plt.show()

names = ['group_a','group_b','group_c']
values = [1,10,100]

plt.figure(figsize=(9,3))
plt.subplot(131)
plt.bar(names,values)
plt.subplot(132)
plt.scatter(names,values)
plt.subplot(133)
plt.plot(names,values)
plt.suptitle('Categorical Plotting')
plt.show()

x = np.arange(10)
y = x
line, = plt.plot(x,y,'-')
line.set_antialiased(False) # turn off antialiasing # 关闭抗锯齿 False不关闭

x1 = x
y1 = x*2
x2 = x*3
y2 = x*4
lines = plt.plot(x1,y1,x2,y2)
#use keyword args
plt.setp(lines,color='r',linewidth=2.0)
#or MATLAB style string value pairs
plt.setp(lines,'color','r','linewidth',2.0)

lines = plt.plot([1,2,3])
plt.setp(lines)

import matplotlib.pyplot as plt
plt.figure(1)
plt.subplot(211)
plt.plot([1,2,3])
plt.subplot(212)
plt.plot([4,5,6])

plt.figure(2)
plt.plot([4,5,6])

plt.figure(1)
plt.subplot(211)
plt.title('Easy as 1,2,3')

t = plt.xlabel('my data',fontsize=14,color='red')

mu,sigma = 100,15
x = mu + sigma*np.random.randn(10000)

# the histogram of the data
n,bins,patches = plt.hist(x,50,density=1,facecolor='g',alpha=0.75)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60,0.025,r'$\mu=100,\ \sigma=15$')
plt.axis([40,160,0,0.03])
plt.grid(True)
plt.show()

plt.title(r'$\sigma_i=15$')

ax = plt.subplot(111)
t = np.arange(0.0,5.0,0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t,s,lw=2)
plt.annotate('local max',xy=(2,1),xytext=(3,1.5),arrowprops=dict(facecolor='black',shrink=0.05))
plt.ylim(-2,2)
plt.show()

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import FuncFormatter 

data = {'Barton LLC':109438.50,'frami, Hills and Schmidt':103569.59,'Fritsch,Russel and Anserson':112214.71,'Jerde-Hilpert':112591.43,'Keeling LLC':100934.30,'Koepp Ltd':137351.96,'Trantow-Barrows':123381.38,'White-Trantow':135841.99,'Will LCC':104437.60}

group_data = list(data.values())
group_names = list(data.keys())
group_mean = np.mean(group_data)

fig,ax = plt.subplots()
ax.barh(group_names,group_data)

print(plt.style.available)
plt.style.use('ggplot')

fig,ax = plt.subplots()
ax.barh(group_names,group_data)
labels = ax.get_xticklabels()

fig,ax = plt.subplots()
ax.barh(group_names,group_data)
labels = ax.get_xticklabels()
plt.setp(labels,rotation=45,horizontalalignment='right')

def currency(x,pos):
    """The two args are the value and tick position"""
    if x>=1e6:
        s = '${:1.1F}M'.format(x*1e-6)
    else:
        s = '${:1.0f}K'.format(x*1e-3)
    return s
formatter =  FuncFormatter(currency)

fig,ax = plt.subplots(figsize=(6,8))
ax.barh(group_names,group_data)
labels = ax.get_xticklabels()
plt.setp(labels,rotation=45,horizontalalignment='right')

ax.set(xlim=[-10000,140000],xlabel='Total Revenue',ylabel='Company',title='Company Revenue')
ax.xaxis.set_major_formatter(formatter)

fig,ax = plt.subplots(figsize=(8,8))
ax.barh(group_names,group_data)
labels = ax.get_xticklabels()
plt.setp(labels,rotation=45,horizontalalignment='right')
ax.axvline(group_mean,ls='--',color='b')

for group in [3,5,8]:
    ax.text(145000,group,'New Company',fontsize=10,verticalalignment='center')
ax.title.set(y=1.05)
ax.set(xlim=[-10000,140000],xlabel='Total Revenue',ylabel='Company',title='Company Revenue')
ax.xaxis.set_major_formatter(formatter)
ax.set_xticks([0,25e3,50e3,75e3,100e3,125e3])
fig.subplots_adjust(right=0.1)
plt.show()

print(fig.canvas.get_supported_filetypes())
fig.savefig('sales.png',transparent=False,dpi=80,bbox_inches='tight')

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl 
plt.style.use('ggplot')
data = np.random.randn(50)

with plt.style.context('dark_background'):
    plt.plot(np.sin(np.linspace(0,2*np.pi)),'r-o')
plt.show()

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.color'] = 'b'
plt.plot(data)

mpl.rc('lines',linewidth=4,color='g')
plt.plot(data)

import matplotlib
matplotlib.matplotlib_fname()


import matplotlib.pyplot as plt 
fig = plt.figure()
ax = fig.add_subplot(2,1,1)# two rows,one columns,first plot

fig2 = plt.figure()
ax2 = fig2.add_axes([0.15,0.1,0.7,0.3])

import numpy as np 
t = np.arange(0.0,1.0,0.01)
s = np.sin(2*np.pi*t)
line, = ax.plot(t,s,color='blue',lw=2)

type(ax.lines)
len(ax.lines)
type(line)

plt.getp(fig)
plt.getp(ax)

a = line.get_alpha()
line.set_alpha(0.5*a)
line.set(alpha=0.5,zorder=2)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_axes([0.1,0.1,0.7,0.3])
print(fig.axes)

for ax in fig.axes:
    ax.grid(True)

fig,ax = plt.subplots()
axis = ax.xaxis
axis.get_ticklocs()

axis.get_ticklabels()
axis.get_ticklines()
axis.get_ticklines(minor=True)

np.random.seed(19680801)

fig,ax = plt.subplots()
ax.plot(100*np.random.rand(20))

formatter = ticker.FormatStrFormatter('$%1.2f')
ax.yaxis.set_major_formatter(formatter)

for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_visible(False)
    tick.label2.set_visible(True)
    tick.label2.set_color('green')
plt.show()

line, = ax.plot([1,2,3],label='Inline label')
ax.legend()

line, = ax.plot([1,2,3])
line.set_label('Label via method')
ax.legend

ax.plot([1,2,3])
ax.legend(['A simple line'])

import numpy as np 
import matplotlib.pyplot as plt 
# Make some fake data
a = b = np.arange(0,3,0.02)
c = np.exp(a)
d = c[::-1]#把a倒序
# Creat plots with pre-defined labels
fig,ax = plt.subplots()
ax.plot(a,c,'k--',label='Model length')
ax.plot(a,d,'k:',label='Data length')
ax.plot(a,c + d,'k',label='Total message length')
legend = ax.legend(loc='upper center',shadow=True,fontsize='x-large')
# Put a nicer background color on the legend
legend.get_frame().set_facecolor('C0')
plt.show()

import matplotlib.pyplot as plt 
import numpy as np 
plt.rcParams['savefig.facecolor']='0.8'

def example_plot(ax,fontsize=12):
    ax.plot([1,2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label',fontsize=fontsize)
    ax.set_ylabel('ylabel',fontsize=fontsize)
    ax.set_title('Title',fontsize=fontsize)
plt.close('all')

fig,ax = plt.subplots()
example_plot(ax,fontsize=24)

fig,ax = plt.subplots()
example_plot(ax,fontsize=24)
plt.tight_layout()

plt.close('all')
fig = plt.figure()

ax1 = plt.subplot(221)
ax2 = plt.subplot(223)
ax3 = plt.subplot(122)

example_plot(ax1)
example_plot(ax2)
example_plot(ax3)

plt.tight_layout()


fig,ax = plt.subplots(constrained_layout=False)
example_plot(ax,fontsize=24)

fig,axs = plt.subplots(2,2,constrained_layout=False)
for ax in axs.flat:
    example_plot(ax)

import matplotlib
import matplotlib.pyplot as plt 

fig  = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

fig.suptitle('bold figure suptitle',fontsize=14,fontweight='bold')
ax.set_title('axes title')
ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')

ax.axis([0,10,0,10])
ax.text(3,8,'boxed italics text in data coords',style='italic',bbox={'facecolor':'red','alpha':0.5,'pad':10})
ax.text(2,6,r'an equation: $E=mc^2$',fontsize=15)

import matplotlib.pyplot as plt 
ax = plt.subplot(111)

t = np.arange(0.0,5.0,0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t,s,lw=2)

ax.annotate('local max',xy=(3,1),xycoords='data',xytext=(0.8,0.95),textcoords='axes fraction',arrowprops=dict(facecolor='black',shrink=0.05),horizontalalignment='right',verticalalignment='top')
plt.ylim(-2,2)
plt.show()

###pad页边距
bbox_props = dict(boxstyle='rarrow,pad=0.3',fc='cyan',ec='b',lw=2)
t = ax.text(0.5,0.5,'Direction',ha='center',va='center',rotation=45,size=15,bbox=bbox_props)
bb = t.get_bbox_patch()
bb.set_boxstyle('rarrow',pad=0.6)

#fancybox list 
import matplotlib.pyplot as plt 
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatch
from matplotlib.patches import FancyBboxPatch

styles = mpatch.BoxStyle.get_styles()
spacing = 1.2
figheight = (spacing*len(styles)+0.5)
fig = plt.figure(figsize=(4/1.5,figheight/1.5))
fontsize = 0.3*72

for i,stylename in enumerate(sorted(styles)):
    fig.text(0.5,(spacing*(len(styles)-i)-0.5)/figheight,stylename,ha='center',size=fontsize,transform=fig.transFigure,bbox=dict(boxstyle=stylename,fc='w',ec='k'))
    
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import matplotlib.pyplot as plt 

plt.rcParams['legend.fontsize']=10
fig = plt.figure()
ax = fig.gca(projection='3d')

theta = np.linspace(-4*np.pi,4*np.pi,100)
z = np.linspace(-2,2,100)
r = z**2+1
x = r*np.sin(theta)
y = r*np.cos(theta)

ax.plot(x,y,z,label='parametric curve')
ax.legend()
plt.show()