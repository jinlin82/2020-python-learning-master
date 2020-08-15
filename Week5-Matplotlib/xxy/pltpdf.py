import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig=plt.figure()
fig.suptitle('No axes on this figure')
fig,ax_lst=plt.subplots(2, 2)

a=pd.DataFrame(np.random.rand(4,5),columns=list('abcde'))
a_asarray=a.values##array
a
b=np.matrix([[1,2],[3,4]])
b_asarray=np.asarray(b)##array

x=np.linspace(0,2,100);x
plt.plot(x,x,label='linear')##线性
plt.plot(x,x**2,label='quadratic')##二次方
plt.plot(x,x**3,label='cubic')##立方
plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Simple Plot")
plt.legend()
plt.show()

plt.ion()
plt.ioff()
plt.plot([1.6,2.7])
plt.show()

plt.plot([1, 2, 3, 4], [1, 4, 9, 16],'ro')
plt.axis([0, 6, 0, 20])##获取[xmin, xmax, ymin, ymax]的列表

t=np.arange(0.,5.,0.2);t
plt.plot(t,t,'r--',t,t**2,'bs',t,t**3,'g^')##s——square正方形
##将多个函数绘制在一张图中


##使用关键字data，字符串'a''b'绘图
data = {'a': np.arange(50),'c': np.random.randint(0, 50, 50),'d': np.random.randn(50)}
data
data['b']=data['a']+10*np.random.randn(50)
data['d']=np.abs(data['d'])*100##abs绝对值
data['b']
data['d']

plt.scatter('a', 'b', c='c', s='d', data=data)##c-color,s-气泡
plt.xlabel('entry a')
plt.ylabel('entry b')

names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]
plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
##上方标题
plt.show()

line, = plt.plot(x, x, '--')
line.set_antialiased(False)

line1,line2 = plt.plot(x,x,'-',x,x**2,'--',lw=3,alpha=0.2)##lw默认为1,alpha为颜色深浅

lines=plt.plot([1,2,3])
plt.setp(lines)

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
plt.figure()
plt.subplot(311)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')##'bo'蓝色不连接，'k'黑色 默认连线
plt.subplot(312)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.subplot(313),plt.plot(t2, np.exp(-t2), 'g--')

plt.figure(1)
plt.subplot(121),plt.plot([1,2,3])##纵轴
plt.subplot(122),plt.plot([4,5,6])

plt.figure(2)
plt.plot([4,5,6])

plt.figure(1)
plt.subplot(121)
plt.title('abc')##默认0-1



mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)##概率密度
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')##r
plt.axis([40, 160, 0, 0.03])##横纵坐标轴范围（plt.xlim/plt.xticks刻度/plt.axvspan/axhspan范围)
plt.grid(True)##坐标系格子显示
plt.show()

plt.title(r'$\sigma_i=15$')##r很重要——它表示该字符串是一个原始字符串，不应将反斜杠视为未使用python


ax = plt.subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)

plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),arrowprops=dict(facecolor='black', shrink=0.05))
plt.ylim(-2, 2)
##注释：xy——要注释的点坐标，xytext——注释所在位置，arrowprops箭头——颜色——长度

from matplotlib.ticker import FuncFormatter

data = {'Barton LLC': 109438.50,'Frami, Hills and Schmidt': 103569.59,'Fritsch, Russel and Anderson': 112214.71,'Jerde-Hilpert': 112591.43,'Keeling LLC': 100934.30,'Koepp Ltd': 103660.54,'Kulas Inc': 137351.96,'Trantow-Barrows': 123381.38,'White-Trantow': 135841.99,'Will LLC': 104437.60}

group_data = list(data.values())
group_names = list(data.keys())
group_mean = np.mean(group_data)

fig,ax=plt.subplots(2,2)

fig,ax=plt.subplots(2,1,figsize=(8,12))##默认为1
ax[0].bar(group_names,group_data)
ax[1].barh(group_names,group_data)

print(plt.style.available)##查看可用样式
plt.style.use('ggplot')

from ggplot import *
ggplot(aes(x=[1,2,3,4]))
BSdata=pd.read_csv('./data/BSdata.csv')
GP=ggplot(aes(x='身高',y='体重'),data=BSdata)
GP+geom_point()+geom_line()
##点+线
ggplot(BSdata,aes(x='支出'))+geom_line(aes(y='身高'))


fig,ax=plt.subplots()
ax.barh(group_names,group_data)
labels=ax.get_xticklabels()
plt.setp(labels, rotation=45,horizontalalignment='right')##setp??


plt.rcParams.update({'figure.autolayout': True})##自动布局

fig, ax = plt.subplots()
ax.barh(group_names, group_data)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')


def currency(x, pos):
    """The two args are the value and tick position"""
    if x >= 1e6:
        s = '${:1.1f}M'.format(x*1e-6)
    else:
        s = '${:1.0f}K'.format(x*1e-3)
    return s


from matplotlib.ticker import FuncFormatter
formatter = FuncFormatter(currency)
##？？
fig, ax = plt.subplots(figsize=(6, 8))
ax.barh(group_names, group_data)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')

ax.set(xlim=[-10000, 140000], xlabel='Total Revenue',ylabel='Company',
title='Company Revenue')
ax.xaxis.set_major_formatter(formatter)##？？定义标签

fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(group_names, group_data)
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')
# Add a vertical line, here we set the style in the function call
ax.axvline(group_mean, ls='--', color='r')
# Annotate new companies
for group in [3, 5, 8]:
    ax.text(145000, group, "New Company", fontsize=10,verticalalignment="center")
# Now we'll move our title up since it's getting a little cramped
ax.title.set(y=1.05)##标题与图片之间的距离
ax.set(xlim=[-10000, 140000], xlabel='Total Revenue',ylabel='Company',title='Company Revenue')
ax.xaxis.set_major_formatter(formatter)##定义标签属性$
ax.set_xticks([0, 25e3, 50e3, 75e3, 100e3, 125e3])
fig.subplots_adjust(right=.1)
# 错误参数 right != 0.1 because left can't >= right???
plt.show()

print(fig.canvas.get_supported_filetypes())
# 我们可以使用figure.Figure.savefig()这一方法将我们的图像存储于硬盘中。
# 2.  注意这里有几个有用的参数:
#     1.  transparent=True 如果保存的格式允许,这个参数会将保存图片的背景变为透明。
#     2.  dpi=80 控制输出图像的分辨率 (每英尺块中像素点的数量)。
#     3.  bbox_inches="tight" 将图形的边界与我们的图匹配。

fig.savefig('sales.png', transparent=False, dpi=80, bbox_inches="tight")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('ggplot')
data = np.random.randn(50)
print(plt.style.available)
plt.style.use(['dark_background', 'presentation'])##
plt.style.use(['dark_background', 'ggplot'])##

with plt.style.context('dark_background'):
plt.plot(np.sin(np.linspace(0, 2 * np.pi)), 'ro-')

x=[1,2,3]
plt.plot(x,x)
plt.show()

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.color'] = 'r'
plt.plot(data)
data

mpl.rc('lines',lw=4,color='g')##颜色无显示？？
plt.plot(data,'y')

import matplotlib
matplotlib.matplotlib_fname()##查看哪个文件路径下的`matplotlibrc`文件被加载了

fig=plt.figure()
ax=fig.add_subplot(211)

plt.subplot(211);plt.plot(x,x)
x=np.array([1,2,3])
y=np.array([4,5,6])

fig,ax=plt.subplots(1,4,figsize=(12,5))
ax[0].plot(x,y)
ax[1].plot(x,y+2)
ax[2].plot(x,y**2)
ax[3].plot(x,y**3)

fig2 = plt.figure()
ax2 = fig2.add_axes([0.15, 0.1, 0.7, 0.3])

t = np.arange(0.0, 1.0, 0.01)
s = np.sin(2*np.pi*t)
line, = ax.plot(t, s, color='blue', lw=2)

plt.plot(t,s,'b--')
plt.figure(figsize=(5,8))
plt.subplot(211),plt.plot(t,s,'b--')
plt.subplot(212),plt.plot(t,s,'b--')

type(ax.lines)
len(ax.lines)
type(line)

## delete lines
del ax.lines[0]
#ax.lines.remove(line)

plt.getp(fig)##查看属性
plt.getp(ax)

a = line.get_alpha()
line.set_alpha(0.5*a)##？？

line.set(alpha=0.5, zorder=2)

fig=plt.figure()
ax1=fig.add_subplot(211)
ax2=fig.add_axes([0.1,0.1,0.7,0.3])
ax3=fig.add_axes([0.3,0.3,0.5,0.5])
fig.axes
for ax in fig.axes:
    ax.grid(True)##加网格


fig,ax=plt.subplots()
axis=ax.xaxis
axis.get_ticklocs()
axis.get_ticklabels()
axis.get_ticklines()
axis.get_ticklines(minor=True)

np.random.seed(123)
fig,ax=plt.subplots()
ax.plot(100*np.random.rand(20))

formatter=ticker.FormatStrFormatter('$%1.2f')##ticker未定义
ax.yaxis.set_major_formatter(formatter)

for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_visible(False)
    tick.label2.set_visible(True)
    tick.label2.set_color('green')


line,=plt.plot([1,2,3],label='Inline label')
plt.legend()

line,= plt.plot([1, 2, 3])
line.set_label('Label via method')
plt.legend()

plt.plot([1, 2, 3])
plt.legend(['A simple line'])

fig,ax=plt.subplots()
ax.plot([1,2,3],label='a line')
plt.legend()



legend((line1, line2, line3), ('label1', 'label2', 'label3'))

a = b = np.arange(0, 3, .02)
c = np.exp(a)
d = c[::-1]
fig, ax = plt.subplots()
ax.plot(a, c, 'k--', label='Model length')
ax.plot(a, d, 'k:', label='Data length')
ax.plot(a, c + d, 'k', label='Total message length')
# legend = ax.legend()
legend = ax.legend(loc='upper center', shadow=True,fontsize='x-large')##中间位置&阴影&大字体
legend.get_frame().set_facecolor('C0')##C0？？

plt.rcParams['savefig.facecolor'] = "0.8"
def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label',fontsize=fontsize)##fontsize
    ax.set_ylabel('y-label',fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

plt.close('all')
# fig, ax = plt.subplots()
# example_plot(ax, fontsize=24)
fig, ax = plt.subplots()
example_plot(ax)

fig, ax = plt.subplots()
example_plot(ax, fontsize=10)##字体大小
plt.tight_layout()##自动调整

plt.close('all')
fig = plt.figure()

ax1 = plt.subplot(221)
ax2 = plt.subplot(223)
ax3 = plt.subplot(122)

example_plot(ax1)
example_plot(ax2)
example_plot(ax3)

plt.tight_layout()##自动调整标签距离位置

fig, ax = plt.subplots(constrained_layout=False)
example_plot(ax, fontsize=24)
fig, ax = plt.subplots(constrained_layout=True)
example_plot(ax,, fontsize=24)##效果相同

fig, axs = plt.subplots(2, 2, constrained_layout=False)
for ax in axs.flat:
    example_plot(ax)

fig, axs = plt.subplots(2, 2, constrained_layout=True)
for ax in axs.flat:
    example_plot(ax)

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85) #（调整图形高度）
# Set titles for the figure and the subplot respectively
fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold') # 添加图标题
ax.set_title('axes title')  #添加轴标题
ax.set_xlabel('xlabel')  # 添加轴标签
ax.set_ylabel('ylabel')  # 添加轴标签
# Set both x- and y-axis limits to [0, 10] instead of default [0, 1]
ax.axis([0, 10, 0, 10]) # 修改轴坐标范围
ax.text(3, 8, 'boxed italics text in data coords', style='italic',bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10}) # 在指定位置添加文本，alpha对应透明度，pad对应图形宽度
ax.text(3, 2, 'unicode: Institut für Festkörperphysik')
ax.text(0.95, 0.01, 'colored text in axes coords',verticalalignment='bottom', horizontalalignment='right',transform=ax.transAxes,color='green', fontsize=15)

ax.plot([2], [1], 'o')
ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),arrowprops=dict(facecolor='black', shrink=0.05)) # shrink箭头长短
plt.show()

plt.getp(ax.texts)##属性列表

ax = plt.subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)
##箭头标注(标注位置/箭头位置/箭头颜色/长短)
ax.annotate('local max', xy=(3, 1),  xycoords='data',xytext=(0.8, 0.95), textcoords='axes fraction',arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='right', verticalalignment='top',)

plt.ylim(-2, 2)
plt.show()


ax=plt.subplot(111)
ax.plot([1,2,3])
bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="r", lw=2)
t=ax.text(1, 2, "Direction", ha="center", va="center", rotation=45,
size=15,bbox=bbox_props)
bb = t.get_bbox_patch()
bb.set_boxstyle("rarrow", pad=0.6)

##什么意思？？Fancybox
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatch
from matplotlib.patches import FancyBboxPatch

styles = mpatch.BoxStyle.get_styles()
spacing = 1.2
figheight = (spacing * len(styles) + .5)
fig = plt.figure(figsize=(4 / 1.5, figheight / 1.5))
fontsize = 0.3 * 72

for i, stylename in enumerate(sorted(styles)):
    fig.text(0.5, (spacing * (len(styles) - i) - 0.5) / figheight, stylename,
              ha="center",
              size=fontsize,
              transform=fig.transFigure,
              bbox=dict(boxstyle=stylename, fc="w", ec="k"))

plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
# Prepare arrays x, y, z
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
ax.plot(x, y, z, label='parametric curve')
ax.legend()

plt.show()