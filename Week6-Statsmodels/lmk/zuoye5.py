# statsmodels_exercise
## 构建数据集

从R中提取数据，建立数据框，提取GDP变量形成Y向量，提取Labor,Kapital,Technology,Energy四个变量加上截距向量构成设计矩阵X。

## 初步查看数据集信息

import pandas as pd
import numpy as np

GDP=pd.read_csv('C:/github_repository/2020-python-learning-master/Week5-Matplotlib/lmk/data/GDP.csv')
GDP.info()
GDP.shape

## 建立线性回归模型

import statsmodels.api as sm
import statsmodels.formula.api as smf
gdp=GDP.dropna() # 剔除含缺失值的行
X=gdp[['Labor','Kapital','Technology','Energy']]
X=sm.add_constant(X)
Y=gdp['GDP']
res=sm.OLS(Y,X).fit()
res.summary()
# Y=-2189.0111-0.0664X1+3.1301X2-69.9267X3+0.2185X4

## 参数估计(估计回归模型的系数矩阵、因变量估计值、回归误差向量)

### 系数矩阵
Bhat=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y);Bhat
### 因变量估计值
Yhat=X.dot(Bhat);Yhat
### 回归误差向量
SST=(Y-Y.mean()).T.dot(Y-Y.mean())
SSE=(Y-Yhat).T.dot(Y-Yhat)
SSR=SST-SSE;SSR

## 多元线性回归函数的拟合优度

Rsq=SSR/SST;Rsq
n=gdp.shape[0]
p=4
MSE=SSE/(n-p-1)
MST=SST/(n-1)
AdjRsq=1-MSE/MST;AdjRsq

## 线性关系显著性检验：F检验（右侧检验）

alpha=0.05
MSR=SSR/p
F_value=MSR/MST;F_value
import scipy.stats as stats
rv=stats.f(4,53)
rv.ppf(0.95) # <F_value 线性关系显著
P_value=1-rv.cdf(F_value);P_value # <alpha=0.05

## 回归系数显著性检验：t检验（双侧检验）

sigmahat=MSE**0.5
c=np.diag(np.linalg.inv(X.T.dot(X)))
se_Bhat=sigmahat*(c**0.5)
t_value=Bhat/se_Bhat;t_value
rvt=stats.t(n-p-1)
t_crivalue=rvt.ppf(0.975);t_crivalue
p_value=(1-rvt.cdf(abs(t_value)))*2;p_value # t值可能为负，要取绝对值

## 回归系数的区间估计

Bhat_CI=np.stack((Bhat-t_crivalue*se_Bhat,Bhat+t_crivalue*se_Bhat),axis=1);Bhat_CI

## 回归模型的预测值

### 点预测

x_new=np.array([1,45000,100000.3,500.81,270000])
x_new.dot(Bhat)

### 区间预测

PI=np.array([x_new.dot(Bhat)-t_crivalue*(1+x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5,x_new.dot(Bhat)+t_crivalue*(1+x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5]);PI  # 个别值预测

CI=np.array([x_new.dot(Bhat)-t_crivalue*(x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5,x_new.dot(Bhat)+t_crivalue*(x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5]);CI  # 平均值预测

np.stack((PI,CI))