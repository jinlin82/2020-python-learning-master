from scipy import stats 
import matplotlib.pyplot as plt
import numpy as np


### 第 1 题： 正态分布
###设 $X \sim N(2, 3^{2})$ ， 求：

###1. $0<X<5$ 的概率
rv=stats.norm(2,3) 
rv5=rv.cdf(5)    #cdf：累计分布函数
rv0=rv.cdf(0)
rv5-rv0

###2. $X$ 0.025 右侧分位点
rv.ppf(0.975)  #ppf:百分点

###3. 画出其概率密度曲线和累计分布函数曲线
x=np.linspace(-6, 10, 100)
plt.plot(x, stats.norm.pdf(x,2,3), label='norm pdf')
plt.plot(x, stats.norm.cdf(x,2,3), label='norm cdf')   #pdf：概率密度函数#pdf(x,loc=0,scale=1)loc对应location即位置，scale可理解为比例 



### 第 2 题： 卡方分布
###设 $X \sim \chi^{2}(5)$ ， 求：

###1. $1<X<5$ 的概率
a=stats.chi2(5)
a5=a.cdf(5)    #cdf：累计分布函数
a1=a.cdf(1)
a5-a1

###2. $X$ 0.025 右侧分位点
a.ppf(0.975)  #ppf:百分点

###3. 画出其概率密度曲线和累计分布函数曲线
x=np.linspace(0,15,100)
plt.plot(x, stats.chi2.pdf(x,5), label='chi2 pdf')
plt.plot(x, stats.chi2.cdf(x,5), label='chi2 cdf')



### 第 3 题： 二项分布
###设 $X \sim B(10, 0.3)$ ， 求：

###1. $X=3$ 的概率
b=stats.binom(10,0.3)
b.pmf(3)   

###2. $X$ 0.025 右侧分位点
b.ppf(0.975)

###3. 画出其概率分布率图和累计分布函数曲线
x=np.arange(11)  #离散情况
plt.plot(x, stats.binom.pmf(x,10,0.3),label='binom pdf')
plt.plot(x, stats.binom.cdf(x,10,0.3),label='binom cdf') 



### 第 4 题： 核密度估计
###设 $X \sim N(2, 3^{2})$ ， 求：

###1. 生成1000个随机数字（np.random)
x=np.random.random(1000)

###2. 使用核方法估计随机数字的密度函数并画出其曲线
import seaborn as sns  
sns.kdeplot(x)   # 绘制核密度分布直方图

###3. 添加 $X \sim N(2, 3^{2})$ 的密度曲线
k=np.linspace(-6, 10, 100)
plt.plot(k, stats.norm.pdf(k,2,3), label='norm pdf')
#如何添加曲线？？？