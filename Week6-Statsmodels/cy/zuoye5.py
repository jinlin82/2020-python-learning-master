### 构建数据集

1. 从R中提取数据，建立数据框，提取GDP变量形成Y向量，提取Labor,Kapital,Technology,Energy四个变量加上截距向量构成设计矩阵X。

### 初步查看数据集信息

import numpy as np 
import pandas as pd
import statsmodels.api as sm 
import statsmodels.formula.api as sfm 
gdp = pd.read_csv('./GDP.csv')
gdp.info()
gdp1 = gdp[['GDP','Labor','Kapital','Technology','Energy']]
gdp2 = gdp1.dropna()
gdp2.isna().sum().sum()==0
gdp2.head()

### 建立线性回归模型

Y = gdp2['GDP']
X = gdp2[['Labor','Kapital','Technology','Energy']]
X = sm.add_constant(X) #在第一列上加上1

##WAY1
lm1 = sm.OLS(Y,X).fit()#模型
lm1.summary()
##WAY2
import statsmodels.formula.api as sfm 
lm2 = sfm.ols('GDP~Labor+Kapital+Technology+Energy',data=gdp2).fit()
lm2.summary()

### 参数估计(估计回归模型的系数矩阵、因变量估计值、回归误差向量)


Bhat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y) # 系数矩阵
Yhat = X.dot(Bhat) # 因变量估计值
SST = (Y-Y.mean()).T.dot(Y-Y.mean())
SSE = (Y-Yhat).T.dot(Y-Yhat)
SSR = SST-SSE # 回归误差向量

### 多元线性回归函数的拟合优度

n = gdp.shape[0]
p = 4
MSE = SSE/(n-p-1)
MST = SST/(n-1) #样本方差

Rsq = SSR/SST #调整前的拟合优度
Adj_Rsq = 1 - MSE/MST #调整后的拟合优度

### 线性关系显著性检验：F检验（右侧检验）

MSR = SSR/p 
F_value = MSR/MSE
import scipy.stats as stats 
rvf = stats.f(p,n-p-1)

####way1

F_critical_value = rvf.ppf(0.95) #软件从左边算起，即0.05
F_value>F_critical_value

####way2

P_value_F = 1 - rvf.cdf(F_value)#cdf是左侧概率，要算右侧概率就要用1-
P_value_F<0.05

### 回归系数显著性检验：t检验（双侧检验）

sigmahat = MSE**0.5
c = np.diag(np.linalg.inv(X.T.dot(X)))
se_Bhat = sigmahat*(c**0.5)
t_value = Bhat/se_Bhat

####way1临界值法
rvt = stats.t(p,n-p-1)
t_crit_value = rvf.ppf(0.975) #软件从左边算起，即0.05
t_value>t_crit_value

####way2
P_value_t = (1 - rvf.cdf(np.abs(t_value)))*2
P_value_t < 0.05

### 回归系数的区间估计

Bhat_CI = np.stack((Bhat - t_crit_value*se_Bhat,Bhat + t_crit_value*se_Bhat),axis=1)

### 回归模型的预测值

x_new = np.array([1.0,76105.0,182340.4,3250.18,324939.0])

#### 点预测

x_new.dot(Bhat)

#### 区间预测


indiv_pre = np.array([x_new.dot(Bhat)-t_crit_value*sigmahat*(1+x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5,x_new.dot(Bhat)+t_crit_value*sigmahat*(1+x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5]) # 个别预测

mean_pre = np.array([x_new.dot(Bhat)-t_crit_value*sigmahat*(x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5,x_new.dot(Bhat)+t_crit_value*sigmahat*(x_new.dot(np.linalg.inv(X.T.dot(X))).dot(x_new.T))**0.5]) # 均值预测

np.stack((indiv_pre,mean_pre))