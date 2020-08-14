# N(1,3^2)
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

rv1=stats.norm(1,3)
rv1.mean()

rv1.cdf(5)-rv1.cdf(2) # P(2<X<5)

rv1.ppf(0.975)

rv1.pdf(2)

xs=np.linspace(-10,12,num=101)
plt.plot(xs,rv1.pdf(xs))

np.random.normal(1,3,100)

rv1.stats() # mean,variance

rv1.rvs(100)

stats.norm.ppf(0.975,1,3)

# t distribution
rv2=stats.t(10)
rv2.mean()

rv2.cdf(5)-rv2.cdf(2)

rv2.ppf(0.975)

rv2.pdf(2)

xs=np.linspace(-10,12,num=101)
plt.plot(xs,rv2.pdf(xs))

np.random.standard_t(10,100)

import pandas as pd
BSdata=pd.read_csv('../Week2-Numpy/data/BSdata.csv',encoding='utf-8')
BSdata.身高.mean()-stats.norm.ppf(0.975)*BSdata.身高.std()/np.sqrt(BSdata.shape[0]) # 置信下限
BSdata.身高.mean()+stats.norm.ppf(0.975)*BSdata.身高.std()/np.sqrt(BSdata.shape[0]) # 置信上限

Z0=(BSdata.体重.mean()-70)/(BSdata.体重.std()/np.sqrt(BSdata.shape[0]))

alpha=0.05
(1-stats.norm.cdf(abs(Z0)))*2 # 假设检验P值

def norm_u_test(X,u0): # 定义一个求P值的函数，X服从正态分布
    n=len(X)
    Z0=(np.mean(X)-u0)/(np.std(X,ddof=1)/n**0.5)
    if n>=30:
        P=(1-stats.norm.cdf(abs(Z0)))*2
    else:
        P=(1-stats.t.cdf(abs(Z0),(n-1)))*2
    return P

norm_u_test(BSdata.体重,70)

# 线性回归
import statsmodels.api as sm
import statsmodels.formula.api as smf

trees=pd.read_csv('../Week5-Matplotlib/lmk/data/trees.csv')
mod=smf.ols('Volume~Girth+Height',data=trees)
res=mod.fit()
res.summary()

# 数据的统计分析
## 随机变量及其分布
### 均匀分布
a=0;b=1;y=1/(b-a)
plt.plot(a,y);plt.hlines(y,a,b)

### 正态分布
#### 标准正态分布
from math import sqrt,pi
x=np.linspace(-4,4,50)
y=1/sqrt(2*pi)*np.exp(-x**2/2)
plt.plot(x,y)

#### 正态分布随机数
np.random.normal(10,4,5)
np.random.normal(0,1,5)

z=np.random.normal(0,1,100)
import seaborn as sns
sns.distplot(z)

#### 正态概率图检验
import pandas as pd
BSdata=pd.read_csv('../Week2-Numpy/data/BSdata.csv',encoding='utf-8')
stats.probplot(BSdata.身高,dist='norm',plot=plt) # 正态概率图
stats.probplot(BSdata.支出,dist='norm',plot=plt)

## 数据分析统计基础
### 统计量
#### 简单随机抽样
np.random.randint(0,2,10)

#### 随机抽取样本号
i=np.random.randint(1,53,6);i
BSdata.iloc[i] # 随机抽取的6个学生信息
BSdata.sample(6) # 直接抽取6个学生的信息

### 统计量的分布
#### 中心极限定理
def norm_sim1(N=1000,n=10): # n样本个数，N抽样次数
    xbar=np.zeros(N)        # 模拟样本均值
    for i in range(N):      
        xbar[i]=np.random.normal(0,1,n).mean() # [0,1]上的标准正态随机数及均值
    sns.distplot(xbar,bins=50) # plt.hist(xbar,bins=50)
    print(pd.DataFrame(xbar).describe().T)
norm_sim1()

def norm_sim1(N=1000,n=10):
    xbar=np.zeros(N)
    for i in range(N):
        xbar[i]=np.random.uniform(0,1,n).mean() # [0,1]上的均匀随机数及均值
    sns.distplot(xbar,bins=50)
    print(pd.DataFrame(xbar).describe().T)
norm_sim1()

#### 均值的t分布
x=np.arange(-4,4,0.1)
yn=stats.norm.pdf(x,0,1)
yt3=stats.t.pdf(x,3)
yt10=stats.t.pdf(x,10)
plt.plot(x,yn,'r-',x,yt3,'b.',x,yt10,'g-.')
plt.legend(['N(0,1)','t(3)','t(10)'])

## 基本统计推断方法
### 参数估计方法
#### 点估计
BSdata.身高.mean()
BSdata.身高.std()

#### 区间估计
def t_interval(b,x):
    a=1-b
    n=len(x)
    ta=stats.t.ppf(1-a/2,n-1);ta
    se=x.std()/sqrt(n)
    return(x.mean()-ta*se,x.mean()+ta*se)
t_interval(0.95,BSdata.身高) # 95%置信区间

### 参数的假设检验
stats.ttest_1samp(BSdata.身高,popmean=166)
stats.ttest_1samp(BSdata.身高,popmean=170)

def ttest_1plot(X,mu=0):
    k=0.1
    df=len(X)-1
    t1p=stats.ttest_1samp(X,popmean=mu)
    x=np.arange(-4,4,k);y=stats.t.pdf(x,df)
    t=abs(t1p[0]);p=t1p[1]
    x1=x[x<=-t];y1=y[x<=-t]
    x2=x[x>=t];y2=y[x>=t]
    print('单样本t检验\t t=%6.3f p=%6.4f'%(t,p))
    print('t置信区间:',stats.t.interval(0.95,len(X)-1,X.mean(),X.std()))
    plt.plot(x,y);plt.hlines(0,-4,4)
    plt.vlines(x1,0,y1,colors='r');plt.vlines(x2,0,y2,colors='r')
    plt.text(-0.5,0.5,'p=%6.4f'% t1p[1],fontsize=15)
ttest_1plot(BSdata.身高,mu=0)

# statsmodels基础
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

## a simple example using ols
dat=sm.datasets.get_rdataset('Guerry',site='C:/github_repository/Rdatasets',pakage='HistData').data
results=smf.ols('Lottery ~ Literacy + np.log(Pop1831)',data=dat).fit()
print(results.summary())

nobs=100
X=np.random.random((nobs,2))
X=sm.add_constant(X)
beta=[1,0.1,0.5]
e=np.random.random(nobs)
y=np.dot(X,beta)+e

results=sm.OLS(y,X).fit()
print(results.summary())

## the datasets pakage
### statsmodels available datasets
dat=sm.datasets.longley.load_pandas()
dat.data
dat.endog # 第一列
dat.exog # 除第一列
dat.endog_name
dat.exog_name
dat.names

### using datasets from stata
auto=sm.datasets.webuse('auto')

### using datasets from R
iris=sm.datasets.get_rdataset('iris',site='C:/github_repository/Rdatasets')
iris.data
print(iris.__doc__) # 查看数据帮助信息

# linear model
import numpy as np
import pandas as pd

trees=pd.read_csv('C:/github_repository/2020-python-learning-master/Week5-Matplotlib/lmk/data/trees.csv')
type(trees)
trees.info()
trees.shape

ones=pd.DataFrame(np.repeat(1,trees.shape[0]))
x12=trees[['Girth','Height']]
X=pd.concat([ones,x12],axis=1)
Y=trees.Volume
Bhat=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

X.T # 属性
X.shape
X.dot(X.T) # 方法
np.linalg.inv() # 函数

## R^2
SST=(Y-Y.mean()).T.dot(Y-Y.mean())
Yhat=X.dot(Bhat)
SSE=(Y-Yhat).T.dot(Y-Yhat)
SSR=SST-SSE
Rsq=SSR/SST
n=trees.shape[0]
p=2
MSE=SSE/(n-p-1)
MST=SST/(n-1) # Y的方差
AdjRsq=1-MSE/MST

## F检验
MSR=SSR/p
F_value=MSR/MSE

import scipy.stats as stats
rv=stats.f(2,28)
rv.ppf(0.95)
rv.cdf(3.340)
P_value=1-rv.cdf(F_value)

rv1=stats.norm()
rv1.ppf(0.975)
rv1.pdf(0)

xs=np.linspace(-3,3,101)
ys=rv1.pdf(xs)
import matplotlib.pyplot as plt
plt.plot(xs,ys)

## t检验
sigmahat=MSE**0.5
c11=np.diag(np.linalg.inv(X.T.dot(X)))[1] # B1hat
se_beta1=sigmahat*(c11**0.5) # B1的标准误
t_value=Bhat[1]/se_beta1
rvt=stats.t(28)
t_crit_value=rvt.ppf(0.975)
p_value=(1-rvt.cdf(t_value))*2

### 以向量形式一次性做所有系数的t检验
c=np.diag(np.linalg.inv(X.T.dot(X)))
se_Bhat=sigmahat*(c**0.5)

t_value=Bhat/se_Bhat
rvt=stats.t(28)
t_crit_value=rvt.ppf(0.975)
p_value=(1-rvt.cdf(t_value))*2

## 区间估计
Bhat_CI=np.stack((Bhat-t_crit_value*se_Bhat,Bhat+t_crit_value*se_Bhat),axis=1)

## 点预测
x_new=np.array([1,15,76])
x_new.dot(Bhat)

## 区间预测
### 个别值的区间预测
np.array([x_new.dot(Bhat)-t_crit_value*(1+x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5,x_new.dot(Bhat)+t_crit_value*(1+x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5])

### 平均值的区间预测
np.array([x_new.dot(Bhat)-t_crit_value*(x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5,x_new.dot(Bhat)+t_crit_value*(x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5])