import pandas as pd 
import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt 

trees=pd.read_csv('trees.csv')
trees.info()
trees.shape 
x=np.repeat(1,trees.shape[0]);x

pd.DataFrame(x)
trees[['Girth','Height']]
X=pd.DataFrame({'a':x,'Girth':trees['Girth'],'Height':trees['Height']})
# m=pd.concat([pd.DataFrame(x),trees[['Girth','Height']]],axis=1)按列排列
Y=trees['Volume']
Bhat=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
##求逆np.linalg.inv
# np.dot()矩阵相乘

dir(X)##查看属性
X.T##属性
X.shape
X.dot(X.T)##方法
X.T.dot(X)
np.linalg.inv(X.T.dot(X))##函数
type(X)

SST=(Y-Y.mean()).T.dot(Y-Y.mean())
yhat=X.dot(Bhat)
SSE=(Y-yhat).T.dot(Y-yhat)
SSR=SST-SSE
Rsq=SSR/SST
r=np.sqrt(Rsq)
n=trees.shape[0]
p=2
MSE=SSE/(n-p-1)
MST=SST/(n-1)##Y的方差
MSR=SSR/p
Y.var()
adjRsq=1-MSE/MST##调整R方

##F检验(线性关系)
F=MSR/MSE

stats.
rv=stats.f(2,28)##F分布（k,n-k-1)
F2=rv.ppf(0.95)
F1=rv.ppf(0.05)
F22=rv.ppf(0.975)
F11=rv.ppf(0.025)
##将F与F1,F2比较 判断是否拒绝原假设(单/双侧检验)
P=1-(rv.cdf(F))
##P<0.05 P值与alpha比较 
rv.cdf(8)##累积概率密度函数
rv.pdf(6)#概率密度函数

rv.cdf(rv.ppf(0.975))
rv1.cdf(rv1.ppf(0.975))

rv1=stats.norm()
rv1.ppf(0.975)##左侧分位数1.96（标准正态分布）
rv1.pdf(0)
rv1.cdf(0)

x1=np.linspace(-5,5,100)
y1=rv1.pdf(x1)
plt.plot(x1,y1)


b=np.random.f(2,28,100)
b.mean()
m=stats.f(2,28)
m.mean()
a=np.random.standard_t(10,1000)
a.mean()
a.var()
n=stats.t(28)
n.mean()
np.random.randn(2,3)##标准正态分布随机数
c=np.random.randn(10000)
c.mean()
d=stats.norm(2,3)##生成标准正态分布
d.mean()
d.var()

# t检验(回归系数)
Bhat
sigmahat=MSE**0.5
c=np.diag(np.linalg.inv(X.T.dot(X)))
se_beta1=(c[1]**0.5)*sigmahat
##1 临界值检验
t=Bhat[1]/se_beta1
rvt=stats.t(n-p-1)
t_crit_value=rvt.ppf(0.975)
t>t_crit_value
##2 P值检验
P=(1-rvt.cdf(t))*2
P<alpha

##矩阵形式
se_beta=(c**0.5)*sigmahat
t_value=Bhat/se_beta
rvt=stats.t(n-p-1)
t_crit_value=rvt.ppf(0.975)

P=(1-rvt.cdf(t))*2

##区间估计
Bhat_CI=np.vstack((Bhat-t_crit_value*se_beta,Bhat+t_crit_value*se_beta))
Bhat_CI=np.stack((Bhat-t_crit_value*se_beta,Bhat+t_crit_value*se_beta),axis=1)##按列排列

np.hstack()


##点预测
Bhat
x_new=np.array([1,15,76])
y_hat=(x_new).dot(Bhat)##向量内积

##区间预测
##个别值
a1=t_crit_value*(1+x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5##+1，范围更大
y1=np.stack((y_hat-a1,y_hat+a1))

##平均值
a2=t_crit_value*(x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5##没有1
y2=np.array([y_hat-a2,y_hat+a2])

###sigmahat???

np.stack((y1,y2),axis=0)

##线性回归
##1
import statsmodels.api as sm 
lm1=sm.OLS(Y,X).fit()##需构建x，y
type(lm1)##线性回归模型
dir(lm1)
lm1.summary()

##2
trees
import statsmodels.formula.api as sfm
lm2=sfm.ols('Volume~Girth+Height',data=trees).fit()
lm2.summary()


##statsmodels
import numpy as np 
import statsmodels.api as sm 
import statsmodels.formula.api as smf 

dat=sm.datasets.get_rdataset('Guerry',site='C:/github_repo/Rdatasets',package='HistData').data ##error
results=smf.ols('Lottery~Literacy+np.log(Pop1831)',data=dat).fit()
print(results.summary())

nobs = 100
X=np.random.random((nobs,2))
X=sm.add_constant(X) #左侧添加一列为1
beta=[1,0.1,0.5]
e=np.random.random(nobs)
y=np.dot(X,beta)+e##X(100,3)beta(3,1)
results=sm.OLS(y,X).fit()
print(results.summary()) ##得出结论


import statsmodels.api as sm 
dir(sm)

from statsmodels.regression.linear_model import OLS,WLS
from statsmodels.tools.tools import rank,add_constant##error??

from statsmodels.datasets import macrodata
from statsmodels.stats import diagnostic

import statsmodels.regression.linear_model as lm 
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi 

patsy.dmatrices("y~x+a+b+a:b",data)

mod=sm.OLS(y,x)#建立模型
res=mod.fit()#模型拟合
res.summary()#得出结论

import statsmodels.api as sm
import statsmodels.tsa.api as tsa
import statsmodels.formula.api as smf


##statsmodels数据集
dat=sm.datasets.longley.load_pandas()
dat.data ##查看数据
dat.endog##第一列
dat.exog ##2-末列
dat.endog_name
dat.exog_name##变量名
dat.names ##全部变量名

##stata数据集
a=sm.datasets.webuse('auto')

iris=sm.datasets.get_rdataset('iris',site="C:/github_repo/Rdatasrts")##error
iris.data ##查看数据
iris.__doc__##数据帮助信息
a.__doc__