### 第 1 题： 正态分布

## 设 $X \sim N(2, 3^{2})$ ， 求：

##1. $0<X<5$ 的概率
import scipy.stats as stats
a=stats.norm(2,3)
a.cdf(5)-a.cdf(0)

##2. $X$ 0.025 右侧分位点
a.ppf(0.025)
a.ppf(0.975)

##3. 画出其概率密度曲线和累计分布函数曲线
import matplotlib.pyplot as plt
import numpy as np
b=np.linspace(-6,10,101)
plt.plot(b,a.pdf(b))
plt.plot(b,a.cdf(b))

### 第 2 题： 卡方分布
##设 $X \sim \chi^{2}(5)$ ， 求：
c=stats.chi2(5)
c.mean()
c.var()
##1. $1<X<5$ 的概率
c.cdf(5)-c.cdf(1)

##2. $X$ 0.025 右侧分位点
c.ppf(0.025)
c.ppf(0.975)

##3. 画出其概率密度曲线和累计分布函数曲线
d=np.linspace(0,20,101)
plt.plot(d,c.pdf(d))
plt.plot(d,c.cdf(d))

### 第 3 题： 二项分布

##设 $X \sim B(10, 0.3)$ ， 求：
m=stats.binom(10,0.3)
m.mean()
m.var()
##1. $X=3$ 的概率
m.cdf(3)
m.cdf(6)

##2. $X$ 0.025 右侧分位点
m.ppf(0.025)
m.ppf(0.975)

##3. 画出其概率分布率图和累计分布函数曲线
x=np.random.randint(0,50,100)
plt.plot(x,c.pdf(x))
plt.plot(x,c.cdf(x))


### 第 4 题： 核密度估计

##设 $X \sim N(2, 3^{2})$ ， 求：
z=stats.norm(2,3)
##1. 生成1000个随机数字（np.random)
y=np.random.random(1000)

##2. 使用核方法估计随机数字的密度函数并画出其曲线
import seaborn as sns
sns.set(color_codes=True)
sns.set_style("white")
y=np.random.random(1000)
sns.kdeplot(y)


##3. 添加 $X \sim N(2, 3^{2})$ 的密度曲线
s=np.linspace(-6,10,101)
plt.plot(s,a.pdf(s))


##4. 


