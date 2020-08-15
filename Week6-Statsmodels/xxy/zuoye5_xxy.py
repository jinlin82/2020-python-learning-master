### 构建数据集

##1. 从R中提取数据，建立数据框，提取GDP变量形成Y向量，提取Labor,Kapital,Technology,Energy四个变量加上截距向量构成设计矩阵X。
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as stats

### 初步查看数据集信息
gdp=pd.read_csv('GDP.csv')
gdp.info()
m=np.repeat(1,gdp.shape[0])
X=pd.DataFrame({'a':m,'Labor':gdp['Labor'],'Kapital':gdp['Kapital'],'Technology':gdp['Technology'],'Energy':gdp['Energy']})[1:]
Y=gdp['GDP'][1:]

# gdp=gdp.dropna()


### 建立线性回归模型
##1
import statsmodels.api as sm 
lm1=sm.OLS(Y,X).fit()
lm1.summary()

##2
import statsmodels.formula.api as sfm
lm2=sfm.ols('GDP~Labor+Kapital+Technology+Energy',data=gdp).fit()
lm2.summary()


### 参数估计(估计回归模型的系数矩阵、因变量估计值、回归误差向量)
Bhat=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
SST=(Y-Y.mean()).T.dot(Y-Y.mean())
yhat=X.dot(Bhat)
SSE=(Y-yhat).T.dot(Y-yhat)
SSR=SST-SSE
re=Y-yhat
n=gdp.shape[0]-1
k=4
MSR=SSR/k 
MSE=SSE/(n-k-1)
MST=SST/(n-1)

### 多元线性回归函数的拟合优度
Rsq=SSR/SST
adj_Raq=1-MSE/MST

### 线性关系显著性检验：F检验（右侧检验）

##1
F=(SSR/k)/(SSE/n-k-1)
rv=stats.f(4,53)
F1=rv.ppf(0.95)
# F>F1拒绝原假设

##2
P=1-rv.cdf(F)
# P<alpha

### 回归系数显著性检验：t检验（双侧检验）

Bhat
sigmahat=MSE**0.5
c=np.diag(np.linalg.inv(X.T.dot(X)))
se_beta=c**0.5*sigmahat
t=Bhat/se_beta
rvt=stats.t(n-k-1)
t_crit_value=rvt.ppf(0.975)
##将t与t_crit_value比较

P_value=(1-rvt.cdf(t))*2

### 回归系数的区间估计
np.stack((Bhat-t_crit_value*se_beta,Bhat+t_crit_value*se_beta),axis=1)


### 回归模型的预测值
#### 点预测
X
x_new=np.array([1,30000,300,15,20000])
y_hat=(x_new).dot(Bhat)

#### 区间预测
##个别值
a1=t_crit_value*sigmahat*(1+x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5
y1=np.array([y_hat-a1,y_hat+a1])
##平均值
a2=t_crit_value*sigmahat*(x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5
y2=np.array([y_hat-a2,y_hat+a2])

np.stack((y1,y2),axis=0)