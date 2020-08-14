## lecture 2 ##

import numpy as np
import pandas as pd

trees=pd.read_csv('trees.csv')
trees.info()

ones=pd.DataFrame(np.repeat(1,trees.shape[0]))
x12=trees[['Girth','Height']]

x=pd.concat([ones,x12],axis=1)
y=trees.Volume

bhat=np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

SST=(y-y.mean()).T.dot(y-y.mean())
yhat=x.dot(bhat)
SSE=(y-yhat).T.dot(y-yhat)
SSR=SST-SSE
n=trees.shape[0]
p=2
MSE=SSE/(n-p-1)
MST=SST/(n-1)
y.var()

AdjRsq=1-MSE/MST

## f 检验
MSR=SSR/p
F_value=MSR/MSE

import scipy.stats as stats
rv=stats.f(2,28)
rv.ppf(0.95)
rv.cdf(3.340)
p_value=1-rv.cdf(F_value)

rv1=stats.norm()
rv1.ppf(0.975)
rv1.pdf(0)

xs=np.linspace(-3,3,101)
ys=rv1.pdf(xs)
import matplotlib.pyplot as plt
plt.plot(xs,ys)

np.random.f(2,28,100)

## t 检验
sigmahat=MSE**0.5
c11=np.diag(np.linalg.inv(x.T.dot(x)))[1]
se_beta1=sigmahat*(c11**0.5)

t_value=bhat[1]/se_beta1
rvt=stats.t(n-p-1)
t_crit_value=rvt.ppf(0.975)
p_v=(1-rvt.cdf(t_value))*2


sigmahat=MSE**0.5
c=np.diag(np.linalg.inv(x.T.dot(x)))
se_bhat=sigmahat*(c**0.5)

t_value=bhat/se_bhat
rvt=stats.t(n-p-1)
t_crit_value=rvt.ppf(0.975)
p_v=(1-rvt.cdf(t_value))*2   ## 向量版本

bhat_CI=np.stack((bhat-t_crit_value*se_bhat,bhat+t_crit_value*se_bhat),axis=-1)

x_new=np.array([1,15,76])

x_new.dot(bhat)

np.stack((np.array([x_new.dot(bhat)-t_crit_value*sigmahat*(1+x_new.dot(np.linalg.inv(x.T.dot(x))).dot(x_new.T))**0.5, x_new.dot(bhat)+t_crit_value*sigmahat*(1+x_new.dot(np.linalg.inv(x.T.dot(x))).dot(x_new.T))**0.5]), np.array([x_new.dot(bhat)-t_crit_value*sigmahat*(x_new.dot(np.linalg.inv(x.T.dot(x))).dot(x_new.T))**0.5,x_new.dot(bhat)+t_crit_value*sigmahat*(x_new.dot(np.linalg.inv(x.T.dot(x))).dot(x_new.T))**0.5]))) ## 个别值和平均值的置信区间

import statsmodels.api as sm
lm1=sm.OLS(y,x).fit()
type(lm1)
dir(lm1)
lm1.summary()

import statsmodels.formula.api as sfm
lm2=sfm.ols('Volume~Girth+Height',data=trees).fit()
lm2.summary() ##不需要构建x或y


## statsmodels ##
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

# dat = sm.datasets.get_rdataset("Guerry","HistData").data

dat = pd.read_csv('Guerry.csv')
results = smf.ols('Lottery~Literacy+np.log(Pop1831)',data=dat).fit()
results.summary()

nobs = 100
X=np.random.random((nobs,2))
X=sm.add_constant(X) ## add a column of ones to an array
beta=[1,0.1,0.5]
e=np.random.random(nobs)
y=np.dot(X,beta) + e

results = sm.OLS(y,X).fit()
results.summary()

mod=sm.OLS(y,X) # describe model
res=mod.fit() # fit model
res.summary() # summarize model

# sm.datasets.datasets_name.load_pandas()
dat=sm.datasets.longley.load_pandas()

dat.data
dat.endog
dat.exog
dat.endog_name
dat.exog_name
dat.names

# using datasets from stata
auto=sm.datasets.webuse('auto')

# using datasets from R
