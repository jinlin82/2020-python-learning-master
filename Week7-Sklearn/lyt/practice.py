### scikit-learn 基础 ###
## 参考资料:sklearn.pdf ##

# sklearn机器学习算法 #
## 分类算法：k-近邻、贝叶斯、逻辑回归、决策树与随机森林
## 回归算法：线性回归、岭回归
## 无监督学习算法：聚类


from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
type(digits)
print(digits.data)
digits.target
digits.images[0]

import matplotlib.pyplot as plt 
plt.figure(1,figsize=(3,3))
plt.imshow(digits.images[-1],cmap=plt.cm.gray_r,interpolation='nearest')
# cmap:将标量数据映射到颜色的Colormap实例或注册的颜色图名称
plt.show()

digits.images.shape  # 三维
data = digits.images.reshape((digits.images.shape[0],-1))
data.shape           # 二维


## 支持向量机

from sklearn import datasets 
digits = datasets.load_digits()

from sklearn import svm   # 支持向量机
clf = svm.SVC(gamma=0.001,C=100.)
# C: 浮点数，默认= 1.0
# gamma:  默认 1 / n_features

## clf = fit(X,y[,sample_weight]: 根据给定的训练数据拟合SVM模型.
clf.fit(digits.data[:-1],digits.target[:-1])
# 使用data除最后一行与target除最后一个元素拟合

## predict(X): 对X中的样本执行分类.
clf.predict(digits.data[-1:]) 


## fit(X,y) ##
# X(n_samples, n_features):样本数为行数，特征数为列数

#example: fit a RandomForestClassifier to data
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[1,2,3],[11,12,13]]  # 2 samples; 3 features
y = [0,1]   # classes of each sample
clf.fit(X,y)    #???结果什么意思


## Transformers and pre-processors/转换和预处理

from sklearn.preprocessing import StandardScaler
X = [[0,15],[1,-10]]
StandardScaler().fit(X).transform(X)
# transform(X): outputs a newly transformed sample matrix X. 
#??? 结果如何理解


## Pipeline/管道

from sklearn.preprocessing import StandardScaler   # 标准化数据
from sklearn.linear_model import LogisticRegression   # 逻辑回归
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split   # 随机划分训练集和测试集
from sklearn.metrics import accuracy_score   # 分类准确率

# create a pipeline object
pipe = make_pipeline(StandardScaler(),LogisticRegression(random_state=0))
# random_state:保证程序每次运行都分割一样的训练集和测试集

# load the iris dataset and split it into train and test sets
X,y = load_iris(return_X_y=True)
# return_X_y: 控制输出数据的结构;=True:将因变量和自变量独立导出
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

# fit the whole pipeline
pipe.fit(X_train,y_train)
accuracy_score(pipe.predict(X_test),y_test)
# accuracy_score: 分类准确率分数


## Model evaluation
### 交叉验证：基本思想是将原始数据(dataset)进行分组,一部分做为训练集(train set),另一部分做为验证集(test set),首先用训练集对分类器进行训练,再利用验证集来测试训练得到的模型(model),以此来做为评价分类器的性能指标。

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression   # 线性回归
from sklearn.model_selection import cross_validate   # 交叉验证

X,y = make_regression(n_samples=1000,random_state=0) 
lr = LinearRegression()

result = cross_validate(lr,X,y)  # 交叉验证；默认5步
result['test_score']    # r_squared score is high because dataset is easy


## parameter searches/参数的搜索

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor   # 随机森林
from sklearn.model_selection import RandomizedSearchCV   # 参数优化
from sklearn.model_selection import train_test_split
from scipy.stats import randint

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# train_test_split: 对数据集进行快速打乱（分为训练集和测试集）

# define the parameter space that will be searched over
param_distributions = {'n_estimators': randint(1, 5),'max_depth': randint(5, 10)}
# params_distributions：参数分布，字典格式
# 'n_estimators': 树的数量
# 'max_depth': 每棵树的最大深度

# now create a searchCV object and fit it to the data
search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0), n_iter=5,param_distributions=param_distributions, random_state=0)
# estimator：要传入的模型
# n_iter：随机寻找参数组合的数量
# params_distributions：参数分布

search.fit(X_train, y_train)
search.best_params_      # 得出最优参数
# the search object now acts like a normal random forest estimator
# with max_depth=9 and n_estimators=4
search.score(X_test, y_test)


## sklearn.linear_model.LinearRegression/线性回归

# 例子1 #
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
reg.score(X, y)         # r_squared
reg.coef_               # 系数
reg.intercept_          # 截距项
reg.predict(np.array([[3, 5]]))

# 例子2 #
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]   # np.newaxis: 插入新维度; 2:取第三列数据

# Split the data into training/testingsets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)
# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)
# The coefficients
print('Coefficients: \n', regr.coef_)    # \n: 换行
# The mean squared error
print('Mean squared error: %.2f'% mean_squared_error(diabetes_y_test, diabetes_y_pred))    # %.2f: 保留两位小数
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'% r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
plt.xticks(());plt.yticks(())
plt.show()


## sklearn.cluster.KMeans 

# 例子1 #
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)   # n_clusters：生成的聚类数
kmeans.labels_    # 聚为同一类的对应标签相同

kmeans.predict([[0, 0], [12, 3]])
kmeans.cluster_centers_      # 聚类中心

# 例子2 #
import numpy as np
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target

estimators = [('k_means_iris_8', KMeans(n_clusters=8)),('k_means_iris_3', KMeans(n_clusters=3)),('k_means_iris_bad_init', KMeans(n_clusters=3,n_init=1, init=' random'))]

fignum = 1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']

for name, est in estimators:           #??此循环报错
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
for name, label in [('Setosa', 0),('Versicolour', 1),('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),X[y == label, 0].mean(),X[y == label, 2].mean() + 2, name,horizontalalignment='center',bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose([1,2,0],y).astype(np.float)    #??单行运行时报错
# 帮助理解：aa=np.choose([4,2,1,3,0],[11,22,33,44,55]);aa
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([]);ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width');ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length');ax.set_title('Ground Truth')
ax.dist = 12

## 训练集、验证集和测试集

# 例子 #
### 注意：该例子没有验证集
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

X, y = datasets.load_iris(return_X_y=True)
X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)        # train_size：若为浮点时，表示验证集占总样本的百分比

X_train.shape, y_train.shape

X_test.shape, y_test.shape

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)


## 交叉验证

# 例子 #
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)         # C: 浮点数
scores = cross_val_score(clf, X, y, cv=5)   # cv：交叉验证生成器或可迭代的次数
scores

from sklearn import metrics
scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')  # scoring：调用的方法
scores


## 5 种交叉验证方式

# 例子 #
import numpy as np

## K-fold
from sklearn.model_selection import KFold

X = ["a", "b", "c", "d"]
kf = KFold(n_splits=2)      # n_splits: 表示划分为几块（至少是2）
for train, test in kf.split(X): 
    print("%s %s" % (train, test))
    
# Repeated K-Fold
from sklearn.model_selection import RepeatedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
random_state = 12883823
rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state= random_state)
for train, test in rkf.split(X):
    print("%s %s" % (train, test))

# LeaveOneOut(): 留一法
from sklearn.model_selection import LeaveOneOut
X = [1, 2, 3, 4]
loo = LeaveOneOut()       
for train, test in loo.split(X):
    print("%s %s" % (train, test))

#  LeavePOut: 测试集留下p个
from sklearn.model_selection import LeavePOut
X = np.ones(4)
lpo = LeavePOut(p=2)     
for train, test in lpo.split(X):
    print("%s %s" % (train, test))

# ShuffleSplit(): 随机排列交叉验证
from sklearn.model_selection import ShuffleSplit
X = np.arange(10)
ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)   # test_size: 代表test集所占比例
for train_index, test_index in ss.split(X):
    print("%s %s" % (train_index, test_index))


## Exhaustive Grid Search

param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},]


## Dummy estimators/虚拟估计量

# 例子 #
### create an imbalanced dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
y[y != 1] = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

### compare the accuracy of SVC and most_frequent
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
# SVC #
clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)
# most_frequent: 预测值是出现频率最高的类别 #
clf = DummyClassifier(strategy='most_frequent', random_state=0)     
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

### change the kernel
clf = SVC(kernel='rbf', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)
## 常用核函数kernel:
# 1.线性核函数kernel='linear'
# 2.多项式核函数kernel='poly'
# 3.径向基核函数kernel='rbf'
# 4.sigmod核函数kernel='sigmod'


## Model persistence/模型持久化

# 例子 #
from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
X, y= datasets.load_iris(return_X_y=True)
clf.fit(X, y)

import pickle          # 保存模型
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])

y[0]

from joblib import dump, load    # joblib更高效
dump(clf, 'filename.joblib')
clf = load('filename.joblib')
clf.predict(X[0:1])


## Validation curve/验证曲线
# 随着超参数设置的改变，模型可能从欠拟合到合适再到过拟合的过程; 验证曲线可以确定用于变化参数值的训练和测试分数

import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge   # Ridge: 岭回归

np.random.seed(0)
X, y = load_iris(return_X_y=True)
indices = np.arange(y.shape[0])
np.random.shuffle(indices)    # np.random.shuffle: 打乱顺序
X, y = X[indices], y[indices]

train_scores, valid_scores = validation_curve(Ridge(), X, y, "alpha",np.logspace(-7, 3, 3),cv=5)
# param_name str: 将会改变的参数名称
# param_range: 将要评估的参数值
# cv: 交叉验证生成器或可迭代
train_scores
valid_scores


## Learning curve/学习曲线
# 一个比较理想的学习曲线图应当是：低偏差、低方差，即收敛且误差小
# 函数作用：对于不同大小的训练集，确定交叉验证训练和测试的分数

# 例子 #
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC

train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv =5)
train_sizes
train_scores
valid_scores


# Visualizations/可视化

# 例子 #
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve   # ROC曲线: 用构图法揭示灵敏性和特效性的相互关系
from sklearn.datasets import load_wine

X,y=load_wine(return_X_y=True)
y = y == 2
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
svc = SVC(random_state=42)
svc.fit(X_train, y_train)
svc_disp = plot_roc_curve(svc, X_test, y_test)  #??如何理解
# plot_roc_curve: 以假正例率（1-特效性,False Positive Rate,FPR）为横坐标,真正例率（灵敏度,True Positive Rate,TPR）为纵坐标,绘制的曲线

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha =0.8)
svc_disp.plot(ax=ax, alpha=0.8)


## Pipeline/管道

# 管道构建例子 #
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA()), ('clf', SVC())]
pipe = Pipeline(estimators)
pipe

from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Binarizer
make_pipeline(Binarizer(), MultinomialNB())


## 获取中间处理步骤: Accessing steps

# 例子 #
pipe.steps[0]  # steps里取第一个tuple
pipe[0]        # steps里第一个tuple的class

pipe['reduce_dim']
pipe.named_steps.reduce_dim is pipe['reduce_dim']

pipe[:1]      # step里取第一个tuple，其余内容保留
pipe[-1:]     # step里取最后一个tuple，其余内容保留

pipe.set_params(clf__C=10)   # 修改个别值

from sklearn.model_selection import GridSearchCV
param_grid = dict(reduce_dim__n_components=[2, 5, 10],clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=param_grid)


## Caching transformers/缓存变压器:避免重复计算

# 例子 #
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
estimators = [('reduce_dim', PCA()), ('clf', SVC())]
cachedir = mkdtemp()     # mkdtemp(): 创建名称唯一的目录
pipe = Pipeline(estimators, memory=cachedir)   # memory: 保存Pipeline中间的“transformer”
pipe

# Clear the cache directory when you don't need it anymore
rmtree(cachedir)    # 清除缓存目录


## Transforming target in regression

# 例子 #
import numpy as np
from sklearn.datasets import load_boston
from sklearn.compose import TransformedTargetRegressor   # 在拟合回归模型之前转换目标y
from sklearn.preprocessing import QuantileTransformer   # 非线性转换
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y=True)
transformer = QuantileTransformer(output_distribution='normal')    # 双峰偏态数据转换成正态分布形式
regressor = LinearRegression()
regr = TransformedTargetRegressor(regressor=regressor,transformer=transformer)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

regr.fit(X_train, y_train)
print('R2 score: {0:.2f}'.format(regr.score(X_test, y_test)))

raw_target_regr = LinearRegression().fit(X_train, y_train)
print('R2 score: {0:.2f}'.format(raw_target_regr.score(X_test, y_test)))


## FeatureUnion: composite feature spaces/特征融合

# 例子 #
from sklearn.pipeline import FeatureUnion   # 将几个transformer对象组合到一个新的transformer
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA 
estimators = [('linear_pca', PCA()), ('kernel_pca', KernelPCA() )]
combined = FeatureUnion(estimators)
combined
combined.set_params(kernel_pca='drop')   # 使用set_params替换单个步骤，并通过设置为' drop '忽略它们


## Feature extraction/特征提取

# 例子 #
measurements = [{'city': 'Dubai', 'temperature': 33.},{'city': 'London', 'temperature': 12.},{'city': 'San Francisco', 'temperature': 18.},]

from sklearn.feature_extraction import DictVectorizer   # 特征向量化
vec = DictVectorizer()     
# DictVectorizer: 对非数字,采用0/1的方式进行量化; 对数值型,维持原值
vec.fit_transform(measurements).toarray()
vec.get_feature_names()


## Standardization/标准化

# 标准化例子 #
from sklearn import preprocessing
import numpy as np
X_train = np.array([[ 1., -1., 2.],[ 2., 0., 0.],[ 0., 1., -1.]])

X_scaled = preprocessing.scale(X_train)   # 沿着某个轴(默认axis=0),标准化数据集，以均值为中心，以分量为单位方差
X_scaled
X_scaled.mean(axis=0)      # 均值为0
X_scaled.std(axis=0)       # 方差为1


scaler = preprocessing.StandardScaler().fit(X_train)   # 本质是生成均值和方差
scaler
scaler.mean_
scaler.scale_
scaler.transform(X_train)

X_test = [[-1., 1., 0.]]
scaler.transform(X_test)    #???


## Scaling features to a range

# 例子
X_train = np.array([[ 1., -1., 2.],[ 2., 0., 0.],[ 0., 1., -1.]])

min_max_scaler = preprocessing.MinMaxScaler()     # 最小最大规范化方法
X_train_minmax = min_max_scaler.fit_transform(X_train)   # 本质是生成min()和max()并导出结果
X_train_minmax

X_test = np.array([[-3., -1., 4.]])
X_test_minmax = min_max_scaler.transform(X_test)
X_test_minmax

min_max_scaler.scale_      #???
min_max_scaler.min_        #???


## Mapping to a Uniform distribution/映射到一个均匀分布

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

quantile_transformer = preprocessing.QuantileTransformer(random_state=0)   # QuantileTransformer: 非参数转换，将数据映射到值在0到1之间的均匀分布
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)
np.percentile(X_train[:, 0], [0, 25, 50, 75, 100])      # 四分位数



## Mapping to a Gaussian distribution/映射到一个高斯分布

pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
X_lognormal = np.random.RandomState(616).lognormal(size=(3, 3))  # lognormal: 对数正态分布
X_lognormal
pt.fit_transform(X_lognormal)

quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
X_trans = quantile_transformer.fit_transform(X)    # fit_transform
quantile_transformer.quantiles_


## 正规化 (Normalization)

# 例子 #
from sklearn import preprocessing
import numpy as np

X = [[ 1., -1., 2.],[ 2., 0., 0.],[ 0., 1., -1.]]
X_normalized_L2 = preprocessing.normalize(X, norm='l2')    # L2范数
X_normalized_L1 = preprocessing.normalize(X, norm='l1')    # L1范数

X_normalized_L1
X_normalized_L2

normalizer = preprocessing.Normalizer().fit(X) # fit does nothing
normalizer

normalizer.transform(X)
normalizer.transform([[-1., 1., 0.]])


## 分类属性编码 Encoding categorical features

# 例子 #???如何理解这个例子里的两种方法
enc = preprocessing.OrdinalEncoder()   # OrdinalEncoder: 将分类特征转化为分类数值
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
enc.transform([['female', 'from US', 'uses Safari']])

enc = preprocessing.OneHotEncoder()   # OneHotEncoder: 可以用来处理有序变量，但对于名义变量，我们只有使用哑变量的方式来处理
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
enc.transform([['female', 'from US', 'uses Safari'],['male', 'from Europe', 'uses Safari']]).toarray ()
enc.categories_

## 网上找的：OneHotEncoder,对于需要transform的数组来说，第一列中的值在categories的相应位置存在的，则为1，不存在，则为0 。以此类推，第N列中的值在第N个categories中存在就为1，不存在就为0。将所有 categories中的返回值以行链接，（相当于np.c_[]函数的作用）返回。
# enc=preprocessing.OneHotEncoder()
# data=[[0,0,3],[1,1,0],[0,2,1],[1,0,2]]
# enc.fit(data)
# enc.transform([[0,1,1]]).toarray()

# 例子 #
genders = ['female', 'male']
locations = ['from Africa', 'from Asia', 'from Europe', 'from US']
browsers = ['uses Chrome', 'uses Firefox', 'uses IE', 'uses Safari']
enc = preprocessing.OneHotEncoder(categories=[genders, locations, browsers])
# Note that for there are missing categorical values
# for the 2nd and 3rd feature
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
enc.transform([['female', 'from Asia', 'uses Chrome']]).toarray ()

X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
drop_enc = preprocessing.OneHotEncoder(drop='first').fit(X)
drop_enc.categories_

drop_enc.transform(X).toarray()


## K-bins discretization/离散化
from sklearn import preprocessing
import numpy as np

X = np.array([[ -3., 5., 15 ],
              [  0., 6., 14 ],
              [  6., 3., 11 ]])
est = preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2], encode='ordinal').fit(X)
# n_bins: 每个特征中分箱的个数
# encode='ordinal': 每个特征的每个箱都被编码为一个整数，返回每一列是一个特征，每个特征下含有不同整数编码的箱的矩阵 

est.transform(X)


## Feature binarization/二值化

# 例子 #
from sklearn import preprocessing
import numpy as np

X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]

binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing
binarizer
binarizer.transform(X)

binarizer = preprocessing.Binarizer(threshold=1.1)   # 阈值threshold=n: 小于等于n的数值转为0, 大于n的数值转为1
binarizer.transform(X)


## Generating polynomial features/多项特征值
# 使用多项式的方法来进行的，如果有a，b两个特征，那么它的2次多项式为（1,a,b,a^2,ab, b^2）

# 例子 #
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3, 2)
X

poly = PolynomialFeatures(2)
poly.fit_transform(X)

X = np.arange(9).reshape(3, 3)
X

poly = PolynomialFeatures(degree=3, interaction_only=True)
# degree：控制多项式的度
# interaction_only： 默认为False，如果指定为True，那么就不会有特征自己和自己结合的项，上面的二次项中没有a^2和b^2
poly.fit_transform(X)


## Custom transformers
import numpy as np
from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(np.log1p, validate=True)
X = np.array([[0, 1], [2, 3]])
transformer.transform(X)


## Univariate feature imputation/单变量特征

# 例子 #
# replace missing values, encoded as np.nan, using the mean
import numpy as np
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit([[1, 2], [np.nan, 3], [7, 6]])

X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))

import pandas as pd
df = pd.DataFrame([["a", "x"],
                   [np.nan, "y"],
                   ["a", np.nan],
                   ["b", "y"]], dtype="category")
imp = SimpleImputer(strategy="most_frequent")
print(imp.fit_transform(df))     #??结果怎么看


## Nearest neighbors imputation/最近邻

# 例子 #
import numpy as np
from sklearn.impute import KNNImputer  # 缺失值插补方法

nan = np.nan
X = [[1, 2, nan], [3, 4, 3], [nan, 6, 5], [8, 8, 7]]
imputer = KNNImputer(n_neighbors=2, weights="uniform")
imputer.fit_transform(X)


## Marking imputed values/标记
from sklearn.impute import MissingIndicator
X = np.array([[-1, -1, 1, 3],
              [4, -1, 0, -1],
              [8, -1, 1, 0]])
indicator = MissingIndicator(missing_values=-1)
mask_missing_values_only = indicator.fit_transform(X)
mask_missing_values_only

indicator.features_


## Gaussian random projection/高斯随机映射
# 通过将原始输入空间投影在随机生成的矩阵上来降低维度

# 例子 #
import numpy as np
from sklearn import random_projection

X = np.random.rand(100, 10000)
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
X_new.shape


## Downloading datasets from the openml.org repository 

import numpy as np
from sklearn.datasets import fetch_openml
mice = fetch_openml(name='miceprotein', version=4)

mice.data.shape
mice.target.shape
np.unique(mice.target)

print(mice.DESCR)
mice.details
mice.url


