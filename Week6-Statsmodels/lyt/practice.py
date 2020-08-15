#### 参考文件：statsmodels.pdf #####

# A simple example using ordinary least squares
import numpy as np
import statsmodels.api as sm  
import statsmodels.formula.api as smf

dat = sm.datasets.get_rdataset("Guerry", site="C:/github_repo/ Rdatasets", package="HistData").data
results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data= dat).fit()
print(results.summary())

nobs = 100
X = np.random.random((nobs, 2))
X = sm.add_constant(X)
beta = [1, .1, .5]
e = np.random.random(nobs)
y = np.dot(X, beta) + e

results = sm.OLS(y, X).fit()
print(results.summary())


# Model fit and summary

mod = sm.OLS(y, X) # Describe model
res = mod.fit() # Fit model
print(res.summary()) # Summarize model


# 例子/statsmodels Available Datasets

import statsmodels.api as sm
dat = sm.datasets.longley.load_pandas()

dat.data
dat.endog
dat.exog
dat.endog_name
dat.exog_name
dat.names


# Using Datasets from Stata

import statsmodels.api as sm

auto=sm.datasets.webuse('auto')


# Using Datasets from R 

import statsmodels.api as sm
iris=sm.datasets.get_rdataset('iris', site="C:/github_repo/ Rdatasets")
iris.data
print(iris.__doc__) 