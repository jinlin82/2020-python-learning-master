# 内置数据集
from sklearn import datasets
iris=datasets.load_iris()
digits=datasets.load_digits()
print(digits.data)
digits.target
digits.images[0]

import matplotlib.pyplot as plt
plt.figure(1,figsize=(3,3))
plt.imshow(digits.images[-1],cmap=plt.cm.gray_r,interpolation='nearest')
plt.show()

# 数据预处理（更改形状）
digits.images.shape
data=digits.images.reshape((digits.images.shape[0],-1))

# 支持向量机
from sklearn import svm
clf=svm.SVC(gamma=0.001,C=100.)

clf.fit(digits.data[:-1],digits.target[:-1])
clf.predict(digits.data[-1:])

# 将 `RandomForestClassifier` 与数据匹配
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(random_state=0)
X=[[1,2,3],
   [11,12,13]] # 2 samples, 3 features
y=[0,1] # classes of each sample
clf.fit(X,y)

# transform
from sklearn.preprocessing import StandardScaler
X=[[0, 15],
   [1,-10]]
StandardScaler().fit(X).transform(X)

# pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

### create a pipeline object
pipe=make_pipeline(StandardScaler(),LogisticRegression(random_state=0))

### load the iris dataset and split it into train and test sets
X,y=load_iris(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

### fit the whole pipeline
pipe.fit(X_train,y_train)
### we can now use it like any other estimator
accuracy_score(pipe.predict(X_test),y_test)

# model evaluation
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

X,y=make_regression(n_samples=1000,random_state=0)
lr=LinearRegression()

result=cross_validate(lr,X,y)
result['test_score']

# parameter searches
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import randint

X,y=fetch_california_housing(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
### define the parameter space that will be searched over
param_distributions={'n_estimators':randint(1,5),'max_depth':randint(5,10)}
### now create a searchCV object and fit it to the data
search=RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),n_iter=5,param_distributions=param_distributions,random_state=0)
search.fit(X_train,y_train)
search.best_params_
### the search object now acts like a normal random forest estimator
### with max_depth=9 and n_estimators=4
search.score(X_test,y_test)

# 线性回归
import numpy as np
from sklearn.linear_model import LinearRegression
X=np.array([[1,1],[1,2],[2,2],[2,3]])
y=np.dot(X,np.array([1,2]))+3
reg=LinearRegression().fit(X,y)
reg.score(X,y)
reg.coef_
reg.predict(np.array([[3,5]]))

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score

### Load the diabetes dataset
diabetes_X,diabetes_y=datasets.load_diabetes(return_X_y=True)
### Use only one feature
diabetes_X=diabetes_X[:,np.newaxis,2]
### Split the data into training/testing sets
diabetes_X_train=diabetes_X[:-20]
diabetes_X_test=diabetes_X[-20:]
### Split the targets into training/testing sets
diabetes_y_train=diabetes_y[:-20]
diabetes_y_test=diabetes_y[-20:]
### Create linear regression object
regr=linear_model.LinearRegression()
### Train the model using the training sets
regr.fit(diabetes_X_train,diabetes_y_train)
### Make predictions using the testing set
diabetes_y_pred=regr.predict(diabetes_X_test)
### The coefficients
print('Coefficients:\n',regr.coef_)
### The mean squared error
print('Mean squared error: %.2f'% mean_squared_error(diabetes_y_test,diabetes_y_pred))
### The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'% r2_score(diabetes_y_test,diabetes_y_pred))
### Plot outputs
plt.scatter(diabetes_X_test,diabetes_y_test,color='black')
plt.plot(diabetes_X_test,diabetes_y_pred,color='blue',linewidth=3)
plt.xticks(());plt.yticks(())
plt.show()

# k-均值聚类
from sklearn.cluster import KMeans
import numpy as np
X=np.array([[1,2],[1,4],[1,0],[10,2],[10,4],[10,0]])
kmeans=KMeans(n_clusters=2,random_state=0).fit(X)
kmeans.labels_
kmeans.predict([[0,0],[12,3]])
kmeans.cluster_centers_


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(5)
iris=datasets.load_iris()
X=iris.data
y=iris.target

estimators=[('k_means_iris_8',KMeans(n_clusters=8)),
            ('k_means_iris_3',KMeans(n_clusters=3)),
            ('k_means_iris_bad_init',KMeans(n_clusters=3,n_init=1,init='random'))]
fignum=1
titles=['8 clusters','3 clusters','3 clusters, bad initialization']
for name,est in estimators:
    fig=plt.figure(fignum,figsize=(4,3))
    ax=Axes3D(fig,rect=[0,0,.95,1],elev=48,azim=134)
    est.fit(X)
    labels=est.labels_

    ax.scatter(X[:,3],X[:,0],X[:,2],c=labels.astype(np.float),edgecolors='k')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum-1])
    ax.dist=12
    fignum=fignum+1

fig=plt.figure(fignum,figsize=(4,3))
ax=Axes3D(fig,rect=[0,0,.95,1],elev=48,azim=134)
for name,label in [('Setosa',0),
                   ('Versicolour',1),
                   ('Virginica',2)]:
    ax.text(X[y==label,3].mean(),
            X[y==label,0].mean(),
            X[y==label,2].mean()+2,name,
            horizontalalignment='center',
            bbox=dict(alpha=.2,edgecolor='w',facecolor='w'))
### Reorder the labels to have colors matching the cluster results
y=np.choose(y,[1,2,0]).astype(np,float)
ax.scatter(X[:,3],X[:,0],X[:,2],c=y,edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
ax.dist=12
fig.show()

# 训练集，验证集和测试集（该例子没有验证集）
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

X,y=datasets.load_iris(return_X_y=True)
X.shape,y.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=0)
X_train.shape,y_train.shape
X_test.shape,y_test.shape
clf=svm.SVC(kernel='linear',C=1).fit(X_train,y_train)
clf.score(X_test,y_test)

# 交叉验证
from sklearn.model_selection import cross_val_score
clf=svm.SVC(kernel='linear',C=1)
scores=cross_val_score(clf,X,y,cv=5)
scores
from sklearn import metrics
scores=cross_val_score(clf,X,y,cv=5,scoring='f1_macro')
scores

# 5种交叉验证方式
from sklearn.model_selection import KFold
X=['a','b','c','d']
kf=KFold(n_splits=2)
for train,test in kf.split(X):
    print("%s %s" % (train,test))

from sklearn.model_selection import RepeatedKFold
X=np.array([[1,2],[3,4],[1,2],[3,4]])
random_state=12883823
rkf=RepeatedKFold(n_splits=2,n_repeats=2,random_state=random_state)
for train,test in rkf.split(X):
    print("%s %s" % (train,test))

from sklearn.model_selection import LeaveOneOut
X=[1,2,3,4]
loo=LeaveOneOut()
for train,test in loo.split(X):
    print("%s %s" % (train,test))

from sklearn.model_selection import LeavePOut
X=np.ones(4)
lpo=LeavePOut(p=2)
for train,test in lpo.split(X):
    print("%s %s" % (train,test))

from sklearn.model_selection import ShuffleSplit
X=np.arange(10)
ss=ShuffleSplit(n_splits=5,test_size=0.25,random_state=0)
for train_index,test_index in ss.split(X):
    print("%s %s" % (train_index,test_index))

# dummy estimators 虚拟估计量
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X,y=load_iris(return_X_y=True)
y[y != 1]=-1
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
clf=SVC(kernel='linear',C=1).fit(X_train,y_train)
clf.score(X_test,y_test)
clf=DummyClassifier(strategy='most_frequent',random_state=0)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

clf=SVC(kernel='rbf',C=1).fit(X_train,y_train)
clf.score(X_test,y_test)

# 模型持久性
from sklearn import svm
from sklearn import datasets
clf=svm.SVC()
X,y=datasets.load_iris(return_X_y=True)
clf.fit(X,y)

import pickle
s=pickle.dumps(clf)
clf2=pickle.loads(s)
clf2.predict(X[0:1])

y[0]

from joblib import dump,load
dump(clf,'filename.joblib')
clf=load('filename.joblib')

# 验证曲线
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge

np.random.seed(0)
X,y=load_iris(return_X_y=True)
indices=np.arange(y.shape[0])
np.random.shuffle(indices)
X,y=X[indices],y[indices]

train_scores,valid_scores=validation_curve(Ridge(),X,y,'alpha',np.logspace(-7,3,3),cv=5)
train_scores
valid_scores

# 学习曲线
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC

train_sizes,train_scores,valid_scores=learning_curve(SVC(kernel='linear'),X,y,train_sizes=[50,80,110],cv=5)

train_sizes
train_scores
valid_scores

# 可视化
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve
from sklearn.datasets import load_wine

X,y=load_wine(return_X_y=True)
y=y==2
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
svc=SVC(random_state=42)
svc.fit(X_train,y_train)
svc_disp=plot_roc_curve(svc,X_test,y_test)

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(random_state=42)
rfc.fit(X_train,y_train)
ax=plt.gca()
rfc_disp=plot_roc_curve(rfc,X_test,y_test,ax=ax,alpha=0.8)
svc_disp.plot(ax=ax,alpha=0.8)

# 管道构建
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
estimators=[('reduce_dim',PCA()),('clf',SVC())]
pipe=Pipeline(estimators)
pipe

from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Binarizer
make_pipeline(Binarizer(),MultinomialNB())

# 获取中间处理步骤
pipe.steps[0]
pipe[0]

pipe['reduce_dim']
pipe.named_steps.reduce_dim is pipe['reduce_dim']

pipe[:1]
pipe[-1:]

pipe.set_params(clf__C=10)

from sklearn.model_selection import GridSearchCV
param_grid=dict(reduce_dim__n_components=[2,5,10],clf__C=[0.1,10,100])
grid_search=GridSearchCV(pipe,param_grid=param_grid)

# 缓存变压器
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
estimators=[('reduce_dim',PCA()),('clf',SVC())]
cachedir=mkdtemp()
pipe=Pipeline(estimators,memory=cachedir)
pipe

rmtree(cachedir) # 当不需要该缓存目录时，即可清除它

# 回归目标转换
import numpy as np
from sklearn.datasets import load_boston
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X,y=load_boston(return_X_y=True)
transformer=QuantileTransformer(output_distribution='normal')
regressor=LinearRegression()
regr=TransformedTargetRegressor(regressor=regressor,transformer=transformer)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
regr.fit(X_train,y_train)

print('R2 score:{0:.2f}'.format(regr.score(X_test,y_test)))

raw_target_regr=LinearRegression().fit(X_train,y_train)
print('R2 score:{0:.2f}'.format(raw_target_regr.score(X_test,y_test)))

# featureunion
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
estimators=[('linear_pca',PCA()),('kernel_pca',KernelPCA())]
combined=FeatureUnion(estimators)
combined

combined.set_params(kernel_pca='drop')

# feature extraction
measurements=[{'city':'Dubai','temperature':33.},
              {'city':'London','temperature':12.},
              {'city':'San Francisco','temperature':18.}] # “城市”是分类属性，“温度”是传统的数值特征
from sklearn.feature_extraction import DictVectorizer
vec=DictVectorizer()

vec.fit_transform(measurements).toarray()
vec.get_feature_names()

# 标准化
from sklearn import preprocessing
import numpy as np
X_train=np.array([[1.,-1., 2.],
                  [2., 0., 0.],
                  [0., 1.,-1.]])
X_scaled=preprocessing.scale(X_train)

X_scaled
X_scaled.mean(axis=0)
X_scaled.std(axis=0)

scaler=preprocessing.StandardScaler().fit(X_train)
scaler
scaler.mean_
scaler.scale_

scaler.transform(X_train)

# scaling features to a range
X_test=[[-1.,1.,0.]]
scaler.transform(X_test)

X_train=np.array([[1.,-1., 2.],
                  [2., 0., 0.],
                  [0., 1.,-1.]])

min_max_scaler=preprocessing.MinMaxScaler()
X_train_minmax=min_max_scaler.fit_transform(X_train)
X_train_minmax

X_test=np.array([[-3.,-1.,4.]])
X_test_minmax=min_max_scaler.transform(X_test)
X_test_minmax

min_max_scaler.scale_
min_max_scaler.min_

# 非线性变换
## 映射到均匀分布
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X,y=load_iris(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
quantile_transformer=preprocessing.QuantileTransformer(random_state=0)
X_train_trans=quantile_transformer.fit_transform(X_train)
X_test_trans=quantile_transformer.transform(X_test)
np.percentile(X_train[:,0],[0,25,50,75,100])

## 映射到高斯分布
pt=preprocessing.PowerTransformer(method='box-cox',standardize=False)
X_lognormal=np.random.RandomState(616).lognormal(size=(3,3))
X_lognormal
pt.fit_transform(X_lognormal)

quantile_transformer=preprocessing.QuantileTransformer(output_distribution='normal',random_state=0)
X_trans=quantile_transformer.fit_transform(X)
quantile_transformer.quantiles_

# 正规化
from sklearn import preprocessing
import numpy as np

X=[[1.,-1., 2.],
   [2., 0., 0.],
   [0., 1.,-1.]]
X_normalized_L2=preprocessing.normalize(X,norm='l2') # axis=1
X_normalized_L1=preprocessing.normalize(X,norm='l1') # axis=0


X_normalized_L1
X_normalized_L2

normalizer=preprocessing.Normalizer().fit(X) # fit does nothing
normalizer

normalizer.transform(X)
normalizer.transform([[-1.,1.,0.]])

# 分类属性编码
enc=preprocessing.OrdinalEncoder() # 将分类特征编码为整数组
X=[['male','from US','uses Safari'],['female','from Europe','uses Firefox']]
enc.fit(X)
enc.transform([['female','from US','uses Safari']])

enc=preprocessing.OneHotEncoder()
X=[['male','from US','uses Safari'],['female','from Europe','uses Firefox']]
enc.fit(X)
enc.transform([['female','from US','uses Safari'],
               ['male','from Europe','uses Safari']]).toarray()

enc.categories_


genders=['female','male']
locations=['from Africa','from Asia','from Europe','from US']
browsers=['uses Chrome','uses Firefox','uses IE','uses Safari']
enc=preprocessing.OneHotEncoder(categories=[genders,locations,browsers])
### 此时存在分类值缺失 
### 对于第二和第三特征
X=[['male','from US','uses Safari'],['female','from Europe','uses Firefox']]
enc.fit(X)
enc.transform([['female','from Asia','uses Chrome']]).toarray()

X=[['male','from US','uses Safari'],['female','from Europe','uses Firefox']]
drop_enc=preprocessing.OneHotEncoder(drop='first').fit(X)
drop_enc.categories_

drop_enc.transform(X).toarray()

# 离散化
from sklearn import preprocessing
import numpy as np

X=np.array([[-3., 5., 15],
            [ 0., 6., 14],
            [ 6., 3., 11]])
est=preprocessing.KBinsDiscretizer(n_bins=[3,2,2],encode='ordinal').fit(X)

est.transform(X)

# feature binarization 特征二值化
from sklearn import preprocessing
import numpy as np

X=[[1.,-1., 2.],
   [2., 0., 0.],
   [0., 1.,-1.]]

binarizer=preprocessing.Binarizer().fit(X) # fit does nothing
binarizer
binarizer.transform(X)

binarizer=preprocessing.Binarizer(threshold=1.1)
binarizer.transform(X)

# generating polynomial features 生成多项式特征
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X=np.arange(6).reshape(3,2)
X

poly=PolynomialFeatures(2)
poly.fit_transform(X)

X=np.arange(9).reshape(3,3)
X

poly=PolynomialFeatures(degree=3,interaction_only=True)
poly.fit_transform(X)

# custom transformers 定制变压器
import numpy as np
from sklearn.preprocessing import FunctionTransformer

transformer=FunctionTransformer(np.log1p,validate=True)
X=np.array([[0,1],[2,3]])
transformer.transform(X)

# 单变量特征插补
## replace missing values, encoded as np.nan, using the mean
import numpy as np
from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.nan,strategy='mean')
imp.fit([[1,2],[np.nan,3],[7,6]])

X=[[np.nan,2],[6,np.nan],[7,6]]
print(imp.transform(X))


import pandas as pd
df=pd.DataFrame([['a','x'],
                 [np.nan,'y'],
                 ['a',np.nan],
                 ['b','y']],dtype='category')

imp=SimpleImputer(strategy='most_frequent')
print(imp.fit_transform(df))

# 近邻插补
import numpy as np
from sklearn.impute import KNNImputer

nan=np.nan
X=[[1,2,nan],[3,4,3],[nan,6,5],[8,8,7]]
imputer=KNNImputer(n_neighbors=2,weights='uniform')
imputer.fit_transform(X)

# marking imputed values 标记估算值
from sklearn.impute import MissingIndicator
X=np.array([[-1, -1, 1, 3],
            [ 4, -1, 0,-1],
            [ 8, -1, 1, 0]])
indicator=MissingIndicator(missing_values=-1)
mask_missing_values_only=indicator.fit_transform(X)
mask_missing_values_only

indicator.features_

# Gaussian random projection
import numpy as np
from sklearn import random_projection

X=np.random.rand(100,10000)
transformer=random_projection.GaussianRandomProjection()
X_new=transformer.fit_transform(X)
X_new.shape

# downloading datasets from the openml.org repository
import numpy as np
from sklearn.datasets import fetch_openml
mice=fetch_openml(name='miceprotein',version=4)

mice.data.shape
mice.target.shape
np.unique(mice.target)

print(mice.DESCR)
mice.details
mice.url