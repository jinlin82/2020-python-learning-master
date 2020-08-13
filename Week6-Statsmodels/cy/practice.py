### statsmodels
import numpy as np 
import statsmodels.api as sm 
import statsmodels.formula.api as smf 

dat = sm.datasets.get_rdataset('Guerry',site='C:/github_repo/Rdatasets',package='HistData').data
##wrong
results = smf.ols('Lottery~Literacy+np.log(Pop1831)',data=dat).fit()
print(results.summary())

nobs = 100
X = np.random.random((nobs,2))
X = sm.add_constant(X) #在第一列上加上1
beta = [1,0.1,0.5]
e = np.random.random(nobs)
y = np.dot(X,beta)+e
 ### 查看一下dot的用法
results = sm.OLS(y,X).fit()
print(results.summary()) # 显示出一个表

## Functions and classes
from statsmodels.regression.linear_model import OLS,WL ### wrong
from statsmodels.tools.tools import rank,add_constant ### wrong

##Modules
from statsmodels.datasets import macrodata
from statsmodels.stats import diagnostic

##Modules with aliases 
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi

##
import patsy # patsy is a package
patsy.dmatrices('y~x+a+b+a:b',data)# a:b means the interation of a and b 

mod = sm.OLS(y,X) # Descibe model
res = mod.fit() # Fit model
print(res.summary()) # Summarize model

#main statsmodels API 
import statsmodels.api as sm 
import statsmodels.tsa.api as tsa
import statsmodels.formula.api as smf

import statsmodels.api as sm 
dat = sm.datasets.longley.load_pandas()

dat.data # 数据类型为数据框
dat.endog #显示data中第一列数据
dat.exog #单变量一般不显示,显示除了第一列的数据框
dat.endog_name #显示data中第一列的名字
dat.exog_name #显示data中除了第一列的名字
dat.names #显示data中所有列名字

#Using Datasets from Stata
import statsmodels.api as sm 
auto = sm.datasets.webuse('auto')###timeout error,它是从哪个网站上搞下来的数据？？？

import statsmodels.api as sm

iris = sm.datasets.get_rdataset('iris',site='C:/github_repo/Rdatasets')
#未知的site?要clone项目后修改路径
iris
print(iris._doc_)

##lect2
import numpy as np 
import pandas as pd 


trees = pd.read_csv('./trees.csv')
trees.info()

ones =  pd.DataFrame(np.repeat(1,trees.shape[0]))
x12 = trees[['Girth','Height']]
type(ones)
type(x12)

X = pd.concat([ones,x12],axis=1)
Y = trees['Volume']

Bhat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
dir(X) #列出所有属性和方法
X.T ##属性
X.shape ##属性
X.dot(X.T) ##方法
np.linalg.inv() ##函数
type(X)

SST = (Y-Y.mean()).T.dot(Y-Y.mean())
Yhat = X.dot(Bhat)
SSE = (Y-Yhat).T.dot(Y-Yhat)
SSR = SST - SSE
Rsq = SSR/SST
n = trees.shape[0]
p = 2
MSE = SSE/(n-p-1)
MST = SST/(n-1) #样本方差

Y.var()

Adj_Rsq = 1 - MSE/MST

MSR = SSR/p 
F_value = MSR/MSE

import scipy.stats as stats 
rv = stats.f(2,28)
rv.ppf(0.95) #软件从左边算起，即0.05
rv.cdf(3.340) #cdf是左侧概率，要算右侧概率就要用1-
P_value = 1 - rv.cdf(F_value)

rv1 = stats.norm()
rv1.ppf(0.975)
rv1.pdf(0)

xs = np.linspace(-3,3,101)
ys = rv1.pdf(xs)
import matplotlib.pyplot as plt 
plt.plot(xs,ys)


### t 检验
sigmahat = MSE**0.5
c11 = np.diag(np.linalg.inv(X.T.dot(X)))[1] #提取第二个
se_beta1 = sigmahat*(c11**0.5)

t_value = Bhat[1]/se_beta1
rvt = stats.t(n-p-1)
t_crit_value = rvt.ppf(0.975)
p_v = (1 - rvt.cdf(t_value))*2

c = np.diag(np.linalg.inv(X.T.dot(X)))
se_Bhat = sigmahat*(c**0.5) #beta的标准误

t_value = Bhat/se_Bhat
rvt = stats.t(n-p-1)
t_crit_value = rvt.ppf(0.975)
p_v = (1-rvt.cdf(t_value))*2

Bhat_CI = np.stack((Bhat - t_crit_value*se_Bhat,Bhat + t_crit_value*se_Bhat),axis=1) #按列


##给出数据来进行预测
x_new = np.array([1,15,76])

##个别预测
a = np.array([x_new.dot(Bhat)-t_crit_value*(1+x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5,x_new.dot(Bhat)+t_crit_value*(1+x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5])

##均值预测
b = np.array([x_new.dot(Bhat)-t_crit_value*(x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5,x_new.dot(Bhat)+t_crit_value*(x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5])

np.stack((np.array([x_new.dot(Bhat)-t_crit_value*(1+x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5,x_new.dot(Bhat)+t_crit_value*(1+x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5]),np.array([x_new.dot(Bhat)-t_crit_value*(x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5,x_new.dot(Bhat)+t_crit_value*(x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5])))

np.stack((a,b))
##第一行个别值，第二行平均值

##WAY1
import statsmodels.api as sm 
lm1 = sm.OLS(Y,X).fit()#模型
type(lm1)
dir(lm1)
lm1.summary()
trees

##WAY2
import statsmodels.formula.api as sfm 
lm2 = sfm.ols('Volume~Girth+Height',data=trees).fit()
lm2.summary()