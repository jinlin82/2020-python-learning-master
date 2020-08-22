from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
print(digits.data)
digits.target #啥意思？
digits.images[0] #啥意思？

#Display the first digit
import matplotlib.pyplot as plt 
plt.figure(1,figsize=(3,3))
plt.imshow(digits.images[-1],cmap=plt.cm.gray_r,interpolation='nearest')
plt.show() #这个图又是什么东西？？

digits.images.shape
data = digits.images.reshape((digits.images.shape[0],-1))

estimator.fit(data)
estimator = Estimator(param1=1,param2=2)
estimator.param1
estimator.estimator_param_
###14-20 wrong 

#支持向量机
from sklearn import datasets
digits = datasets.load_digits()

from sklearn import svm 
clf = svm.SVC(gamma=0.001,C=100.)

clf.fit(digits.data[:-1],digits.target[:-1]) #删除最后一个样本
clf.predict(digits.data[-1:]) #预测最后一个样本

## fit a RandomForestClassifier to data
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[1,2,3],[11,12,13]]
y = [0,1] # classes of each sample 
clf.fit(X,y)

from sklearn.preprocessing import StandardScaler
X = [[0,15],[1,-10]]
StandardScaler().fit(X).transform(X)#???

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline  
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# creat a piprline object

pipe = make_pipeline(StandardScaler(),LogisticRegression(random_state=0))
 #fit the whole pipeline 
 pipe.fit(X_train,y_train)
 #we can now use it like any other estimator
 accuracy_score(pipe.predict(X_test),y_test)

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import cross_validate

X,y = make_regression(n_samples=1000,random_state=0)
lr = LinearRegression()

result = cross_validate(lr,X,y) #defaults to 5-fold CV
result['test_score'] # r_square score is high because dataset is easy

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import randint 
X,y = fetch_california_housing(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0) 

param_distributions = {'n_estimators':randint(1,5),'max_depth':randint(5,10)}

search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),n_iter=5,param_distributions=param_distributions,random_state=0)
search.fit(X_train,y_train)
search.best_params_
search.score(X_test,y_test)

import numpy as np 
from sklearn.linear_model import LinearRegression 
X = np.array([[1,1],[1,2],[2,2],[2,3]])
# y = 1*x_0 + 2*x_1 + 3
y = np.dot(X,np.array([1,2]))+3
reg = LinearRegression().fit(X,y)

reg.score(X,y)
reg.coef_
reg.intercept_
reg.predict(np.array([[3,5]]))

import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score

#load the diabetes dataset
diabetes_X,diabetes_y = datasets.load_diabetes(return_X_y=True)
# Use only one feature
diabetes_X = diabetes_X[:,np.newaxis,2]
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# creat linear regression object
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train,diabetes_y_train)
diabetes_y_pred = regr.predict(diabetes_X_test)

print('Coefficients:\n',regr.coef_)
print('Mean square error:%.2f'% mean_squared_error(diabetes_y_test,diabetes_y_pred))
print('Coefficient of determination:%.2f'% r2_score(diabete_y_test,diabetes_y_pred))

plt.scatter(diabetes_X_test,diabetes_y_test,color='black')
plt.plot(diabetes_X_test,diabetes_y_pred,color='blue',linewidth=3)
plt.xticks(());plt.yticks(())
plt.show()

from sklearn.cluster import KMeans 
import numpy as np 
X = np.array([[1,2],[1,4],[1,0],[10,2],[10,4],[10,0]])
kmeans = KMeans(n_clusters=2,random_state=0).fit(X)
kmeans.labels_
kmeans.predict([[0,0],[12,3]])
kmeans.cluster_centers_

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target

estimators = [('k_means_iris_8',KMeans(n_clusters=8)),('k_means_iris_3',KMeans(n_clusters=3)),('k_means_iris_bad_init',KMeans(n_clusters=3,n_init=1,init='random'))]

fignum = 1
titles = ['8 clusters','3 clusters','3 clusters,bad initialization']
for name,est in estimators:
    fig = plt.figure(fignum,figsize=(4,3))
    ax = Axes3D(fig,rect=[0,0,0.95,1],elev=48,azim=134)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:,3],X[:,0],X[:,2],c=labels.astype(np.float),edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Petal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum-1])
    ax.dist = 12
    fignum = fignum + 1

fig = plt.figure(fignum,figsize=(5,6))
ax = Axes3D(fig,rect=[0,0,0.95,1],elev=48,azim=134)
# rect=[left, bottom, width, height],elev仰角视角默认30,azim方位视角默认-60 
for name,label in [('Setosa',0),('Versicolour',1),('Virginica',2)]:
    ax.text3D(X[y == label,3].mean(),X[y == label,0].mean(),X[y == label,2].mean()+2,name,horizontalalignment='center',bbox=dict(alpha=0.2,edgecolor='w',facecolor='w'))
    
#y = np.choose(y,[1,2,0]).astype(np.float)
ax.scatter(X[:,3],X[:,0],X[:,2],c=y,edgecolors='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
ax.dist = 12 #?
### fig尺寸没设置好的话，zlabel可能就没办法显示出来

#伪代码
# split data 
data = ...
train,validation,test = split(data)

# tune model hyperparameters
parameters = ...
for params in parameters:
    model = fit(train,param)
    skill = evaluate(model,validation)

# evaluate final model for comparision with other models

model = fit(train)
skill = evaluate(model,test)

#该子集没有验证集
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import datasets 
from sklearn import svm 

X,y = datasets.load_iris(return_X_y=True)
X.shape,y.shape

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=0)#如果 random_state = None (默认值），会随机选择一个种子，这样每次都会得到不同的数据划分。

X_train.shape,y_train.shape
X_test.shape,y_test.shape
clf = svm.SVC(kernel='linear',C=1).fit(X_train,y_train)
clf.score(X_test,y_test)

#交叉验证伪代码

# split data 
data = ...
train,test = split(data)

# tune model hyperparameters
parameters = ...
k = ...
for params in parameters:
    skills = list()
    for i in k:
        fold_train,fold_val = cv_split(i,k,train)
        model = fit(fold_train,params)
        skill_estimate = evaluate(model,fold_val)
        skills.append(skill_estimate)
    skill = summarize(skills)

# evaluate final model for comparision 
model = fit(train)
skill = evaluate(model,test)

from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear',C=1)
scores = cross_val_score(clf,X,y,cv=5)
scores

from sklearn import metrics
scores = cross_val_score(clf,X,y,cv=5,scoring='f1_macro')
scores

import numpy as np 

##K-fold 
from sklearn.model_selection import KFold

X = ['a','b','c','d']
kf = KFold(n_splits=2)
for train,test in kf.split(X):
    print('%s %s' % (train,test))

from sklearn.model_selection import LeaveOneOut
X = [1,2,3,4]
loo = LeaveOneOut() #默认留1个
for train,test in loo.split(X):
    print('%s %s' % (train,test))

from sklearn.model_selection import LeavePOut
X = np.ones(4)
lpo = LeavePOut(p=2) # 取2最后一组
for train,test in lpo.split(X):
    print('%s %s' % (train,test))

from sklearn.model_selection import ShuffleSplit
X = np.arange(10)
ss = ShuffleSplit(n_splits=5,test_size=0.25,random_state=0)
for train_index,test_index in ss.split(X):
    print('%s %s' % (train_index,test_index))

estimator.get_params()

param_grid = [{'C':[1,10,100,1000],'kernel':['linear']},{'C':[1,10,100,1000],'gamma':[0.001,0.0001],'kernel':['rbf']}]


from scipy.stats import expon
{'C':expon(scale=100),'gamma':expon(scale=0.1),'kernel':['rbf'],'class_weight':['balanced',None]}

### create an imbalanced dataset
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split 
X,y = load_iris(return_X_y=True)
y[y!=1] = -1
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

### compare the accuracy of SVC and the most_frequent
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
clf = SVC(kernel='linear',C=1).fit(X_train,y_train)
clf.score(X_test,y_test)
clf = DummyClassifier(strategy='most_frequent',random_state=0)
clf.fit(X_train,y_train) 
clf.score(X_test,y_test) 

### change the kernel
clf = SVC(kernel='rbf',C=1).fit(X_train,y_train)
clf.score(X_test,y_test) 

from sklearn import svm 
from sklearn import datasets
clf = svm.SVC()
X,y = datasets.load_iris(return_X_y=True)
clf.fit(X,y)

import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])

y[0]

from joblib import dump,load 
dump(clf,'filename.joblib')
clf = load('filename.joblib')

import numpy as np
from sklearn.model_selection import validation_curve 
from sklearn.datasets import load_iris 
from sklearn.linear_model import Ridge 

np.random.seed(0)
X,y = load_iris(return_X_y=True)
indices = np.arange(y.shape[0])
np.random.shuffle(indices) # 打乱顺序
X,y = X[indices],y[indices]

train_scores,valid_scores = validation_curve(Ridge(),X,y,'alpha',np.logspace(-7,3,3),cv=5)

train_scores
valid_scores


from sklearn.model_selection import learning_curve 
from sklearn.svm import SVC

train_sizes,train_scores,valid_scores = learning_curve(SVC(kernel='linear'),X,y,train_sizes=[50,80,110],cv=5)

train_sizes
train_scores
valid_scores

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve
from sklearn.datasets import load_wine 
X,y = load_wine(return_X_y=True)
y = y == 2 
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)
svc = SVC(random_state=42)
svc.fit(X_train,y_train)
svc_disp = plot_roc_curve(svc,X_test,y_test)

import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier 

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train,y_train)
ax = plt.gca()
rfc_disp = plot_roc_curve(rfc,X_test,y_test,ax=ax,alpha=0.8)

### 构建管道例子
from sklearn.pipeline import Pipeline 
from sklearn.svm import SVC
from sklearn.decomposition import PCA
estimators = [('reduce_dim',PCA()),('clf',SVC())]
pipe = Pipeline(estimators)
pipe 

from sklearn.pipeline import make_pipeline 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.preprocessing import Binarizer 
make_pipeline(Binarizer(),MultinomialNB())

pipe[0]

pipe['reduce_dim']
pipe.named_steps.reduce_dim is pipe['reduce_dim']

pipe[:1]
pipe[-1:]

pipe.set_params(clf__C=10) ##clf__C 中的是两个下划线__ 而不是一个_

from sklearn.model_selection import GridSearchCV
param_grid = dict(reduce_dim_n_components=[2,5,10],clf__C=[0.1,10,100])
grid_search = GridSearchCV(pipe,param_grid=param_grid)

from tempfile import mkdtemp 
from shutil import rmtree 
from sklearn.decomposition import PCA 
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline 
estimators = [('reduce_dim',PCA()),('clf',SVC())]
cachedir = mkdtemp()
pipe = Pipeline(estimators,memory=cachedir)
pipe 
# clear the cache directory when you do not need it any more
rmtree(cachedir)

import numpy as np 
from sklearn.datasets import load_boston 
from sklearn.compose import TransformedTargetRegressor 
from sklearn.preprocessing import QuantileTransformer 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
X,y = load_boston(return_X_y=True)
transformer = QuantileTransformer(output_distribution='normal')
regressor = LinearRegression()
regr = TransformedTargetRegressor(regressor=regressor,transformer=transformer)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
regr.fit(X_train,y_train)

print('R2 score:{0:.2f}'.format(regr.score(X_test,y_test)))

raw_target_regr = LinearRegression().fit(X_train,y_train)
print('R2 score:{0:.2f}'.format(raw_target_regr.score(X_test,y_test)))

from sklearn.pipeline import FeatureUnion 
from sklearn.decomposition import PCA 
from sklearn.decomposition import KernelPCA
estimators = [('linear_pca',PCA()),('kernel_pca',KernelPCA())]
combined = FeatureUnion(estimators)
combined 

combined.set_params(kernel_pca='drop')

measurements = [{'city':'Dubai','temperature':33.},{'city':'London','temperature':12.},{'city':'San Francisco','temperature':18.}]

from sklearn.feature_extraction import DictVectorizer 
vec = DictVectorizer()

vec.fit_transform(measurements).toarray()
vec.get_feature_names()

from sklearn import preprocessing 
import numpy as np 
X_train = np.array([[1.,-1.,2.],[2,0,0],[0,1,-1]])
X_scaled = preprocessing.scale(X_train)

X_scaled
X_scaled.mean(axis=0)
X_scaled.std(axis=0)

scaler = preprocessing.StandardScaler().fit(X_train)
scaler
scaler.mean_
scaler.scale_

scaler.transform(X_train)

X_test = [[-1,1,0]]
scaler.transform(X_test)

X_std = (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
X_scaled = X_std*(X.max(axis=0)-X.min(axis=0))+X.min(axis=0)

X_train = np.array([[1,-1,2],[2,0,0],[0,1,-1]])

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_train_minmax 

X_test = np.array([[-3,-1,4]])
X_test_minmax = min_max_scaler.transform(X_test)
X_test_minmax 

min_max_scaler.scale_
min_max_scaler.min_

from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 

X,y = load_iris(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)
np.percentile(X_train[:,0],[0,25,50,75,100])

pt = preprocessing.PowerTransformer(method='box=cox',standardize=False)
X_lognormal = np.random.RandomState(616).lognormal(size=(3,3))
X_lognormal
pt.fit_transform(X_lognormal) ################# wrong

quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal',random_state=0)
X_trans = quantile_transformer.fit_transform(X)
quantile_transformer.quantiles_

from sklearn import preprocessing 
import numpy as np 
X = [[1,-1,2],[2,0,0],[0,1,-1]]
X_normalized_L2 = preprocessing.normalize(X,norm='l2')
X_normalized_L1 = preprocessing.normalize(X,norm='l1')

X_normalized_L1
X_normalized_L2

normalizer = preprocessing.Normalizer().fit(X)
normalizer

normalizer.transform(X)
normalizer.transform([[-1,1,0]])

enc = preprocessing.OneHotEncoder()
X = [['male','from US','uses Safari'],['female','from Europe','uses Firefox']]
enc.fit(X)
enc.transform([['female','from US','uses Safari']])

enc = preprocessing.OneHotEncoder()
X = [['male','from US','uses Safari'],['female','from Europe','uses Firefox']]
enc.fit(X)
enc.transform([['female','from US','uses Safari'],['male','from Europe','uses Safari']]).toarray()
enc.categories_

genders = ['female','male']
locations = ['from Africa','from Asia','from Europe','from US']
browsers = ['uses Chrome','uses Firefox','uses IE','uses Safari']
enc = preprocessing.OneHotEncoder(categories=[genders,locations,browsers])

#Note that for there are missing categorical values
# for the 2nd and 3rd feature
X = [['male','from US','uses Safari'],['female','from Europe','uses Firefox']]
enc.fit(X)
enc.transform([['female','from Asia','uses Chrome']]).toarray()

X = [['male','from US','uses Safari'],['female','from Europe','uses Firefox']]
drop_enc = preprocessing.OneHotEncoder(drop='first').fit(X)
drop_enc.categories_

drop_enc.transform(X).toarray()

from sklearn import preprocessing 
import numpy as np 

X = np.array([[-3,5,15],[0,6,14],[6,3,11]])
est = preprocessing.KBinsDiscretizer(n_bins=[3,2,2],encode='ordinal').fit(X)
est.transform(X)

from sklearn import preprocessing 
import numpy as np 
X = [[1,-1,2],[2,0,0],[0,1,1]]
binarizer = preprocessing.Binarizer().fit(X)
binarizer
binarizer.transform(X)

binarizer = preprocessing.Binarizer(threshold=1.1)
binarizer.transform(X)

import numpy as np 
from sklearn.preprocessing import PolynomialFeatures 
X = np.arange(6).reshape(3,2)
X

poly = PolynomialFeatures(2)
poly.fit_transform(X)

X = np.arange(9).reshape(3,3)
X

poly = PolynomialFeatures(degree=3,interaction_only=True)
poly.fit_transform(X)

import numpy as np 
from sklearn.preprocessing import FunctionTransformer 

transformer = FunctionTransformer(np.log1p,validate=True)##np.log1p=log(x+1)
X = np.array([[0,1],[2,3]])
transformer.transform(X)

# replace missing values,encoded as np.nan,using the mean
import numpy as np 
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan,strategy='mean')
imp.fit([[1,2],[np.nan,3],[7,6]])

X = [[np.nan,2],[6,np.nan],[7,6]]
print(imp.transform(X))

import pandas as pd 
df = pd.DataFrame([['a','x'],[np.nan,'y'],['a',np.nan],['b','y']],dtype='category')
imp = SimpleImputer(strategy='most_frequent')
print(imp.fit_transform(df))

import numpy as np 
from sklearn.impute import KNNImputer 

##nan用相邻两项平均值来补上
nan = np.nan 
X = [[1,2,nan],[3,4,3],[nan,6,5],[8,8,7]]
imputer = KNNImputer(n_neighbors=2,weights='uniform')
imputer.fit_transform(X)

from sklearn.impute import MissingIndicator
X = np.array([[-1,-1,1,3],[4,-1,0,-1],[8,-1,1,0]])
indicator = MissingIndicator(missing_values=-1)
mask_missing_values_only = indicator.fit_transform(X)
mask_missing_values_only

indicator.features_

import numpy as np 
from sklearn import random_projection

X = np.random.rand(100,10000)
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
X_new.shape

import numpy as np 
from sklearn.datasets import fetch_openml 
mice = fetch_openml(name='miceprotein',version=4)

mice.data.shape 
mice.target.shape 
np.unique(mice.target)

print(mice.DESCR)