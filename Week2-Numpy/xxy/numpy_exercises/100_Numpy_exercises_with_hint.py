
# 100 numpy exercises with hint

This is a collection of exercises that have been collected in the numpy mailing
list, on stack overflow and in the numpy documentation. I've also created some
to reach the 100 limit. The goal of this collection is to offer a quick
reference for both old and new users but also to provide a set of exercises for
those who teach.

#### 1. Import the numpy package under the name `np` (★☆☆) 

(**hint**: import … as …)
import numpy as np


#### 2. Print the numpy version and the configuration (★☆☆) 

(**hint**: np.\_\_version\_\_, np.show\_config)

np.__version__
np.show_config()

#### 3. Create a null vector of size 10 (★☆☆) 

(**hint**: np.zeros)
np.zeros(10)

#### 4.  How to find the memory size of any array (★☆☆) 

(**hint**: size, itemsize)
a=np.random.randint(1,10,(2,3))
np.size(a)
a.itemsize

x=np.zeros((10,10))
"%d bytes"%(x.size*x.itemsize)
x.size
x.itemsize
#### 5.  How to get the documentation of the numpy add function from the command line? (★☆☆) 

(**hint**: np.info)
np.info(a)

np.info(np.add)

#### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆) 

(**hint**: array\[4\])
x=np.zeros(10)
x[4]=1
x

#### 7.  Create a vector with values ranging from 10 to 49 (★☆☆) 

(**hint**: np.arange)

np.arange(10,50)

??#### 8.  Reverse a vector (first element becomes last) (★☆☆) 

(**hint**: array\[::-1\])
a=np.arange(20)
a=a[::-1]
a
#### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆) 

(**hint**: reshape)

np.arange(9).reshape(3,3)

??#### 10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆) 

(**hint**: np.nonzero)
x=[1,2,0,0,4,0]
np.nonzero(x)

y=np.nonzero([1,2,0,0,4,0])
#### 11. Create a 3x3 identity matrix (★☆☆) 

(**hint**: np.eye)
np.eye(3)


#### 12. Create a 3x3x3 array with random values (★☆☆) 

(**hint**: np.random.random)
np.random.random([3,3,3])
np.random.random([3,3])
np.random.random(3)


#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆) 

(**hint**: min, max)
a=np.random.random([10,10])
np.amin(a)
np.amax(a)

a.min()
a.max()
#### 14. Create a random vector of size 30 and find the mean value (★☆☆) 

(**hint**: mean)
x=np.random.rand(30)
np.mean(x)

??#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆) 

(**hint**: array\[1:-1, 1:-1\])
a=np.ones((10,10))
a[1:-1,1:-1]=0
a
#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆) 

(**hint**: np.pad)
x=np.ones((5,5))
np.pad(x,pad_width=1,mode='constant',constant_values=0)
np.pad(x,pad_width=3,mode='constant',constant_values=2)

#### 17. What is the result of the following expression? (★☆☆) 

(**hint**: NaN = not a number, inf = infinity)


```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3*0.1
```
5==2+3
5.0==2+3

#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆) 

(**hint**: np.diag)

m=np.tril([1,2,3,4,5],-1)
m

np.diag(1+np.arange(4),k=-1)

#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆) 

(**hint**: array\[::2\])
x=np.arange(64).reshape(8,8)

x=np.zeros((8,8),dtype=int)
x[1::2,::2]=1
x[::2,1::2]=1
x
#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? 

(**hint**: np.unravel\_index)

np.unravel_index(100,(6,7,8))

#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆) 

(**hint**: np.tile)
x=np.arange(64).reshape(8,8)

np.tile(np.array([[0,1],[1,0]]),(4,4))


#### 22. Normalize a 5x5 random matrix (★☆☆) 

(**hint**: (x - mean) / std)

x=np.random.randint(0,10,[5,5])
np.mean(x)
y=(x-np.mean(x))/np.std(x)
y.mean()
y.var()

z=np.random.random((5,5))
a=(z-z.min())/(z.max()-z.min())
a.mean()
a.var()
??#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆) 

(**hint**: np.dtype)
color=np.dtype([("r",np.ubyte,1),("g",np.ubyte,1),("b",np.ubyte,1),("a",np.ubyte,1)])


#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆) 

(**hint**: np.dot | @)

np.dot(np.ones((5,3)),np.ones((3,2)))

#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆) 

(**hint**: >, <=)
x=np.arange(10)
x[(x>3)&(x<8)]=x[(x>3)&(x<8)]*(-1)
x

Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1
print(Z)
#### 26. What is the output of the following script? (★☆☆) 

(**hint**: np.sum)


```python
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```
range(5)
sum(range(5),0)

#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)

Z=-5
```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```



#### 28. What are the result of the following expressions?


```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```
nan
0
array([-2.14748365e+09])
##/ 代表浮点数除法的，得到的结果是浮点数；// 是整数除法，得到的结果是整数。

#### 29. How to round away from zero a float array ? (★☆☆) 

(**hint**: np.uniform, np.copysign, np.ceil, np.abs)

Z = np.random.uniform(-10,+10,10)
print (np.copysign(np.ceil(np.abs(Z)), Z))

#### 30. How to find common values between two arrays? (★☆☆) 

(**hint**: np.intersect1d)
x=np.random.randint(0,10,(3,3));x
y=np.random.randint(0,10,(3,3));y
np.intersect1d(x,y)

x=np.random.randint(0,10,10);x
y=np.random.randint(0,10,10);y
np.intersect1d(x,y)
#### 31. How to ignore all numpy warnings (not recommended)? (★☆☆) 

(**hint**: np.seterr, np.errstate)

defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0
_ = np.seterr(**defaults)

#### 32. Is the following expressions true? (★☆☆) 

(**hint**: imaginary number)


```python
np.sqrt(-1) == np.emath.sqrt(-1)
```

#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆) 

(**hint**: np.datetime64, np.timedelta64)
np.datetime64
np.timedelta64
yesterday=np.datetime64('today','D')-np.timedelta64(1,'D')
today=np.datetime64('today','D')
tomorrow=np.datetime64('today','D')+np.timedelta64(1,'D')

#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆) 

(**hint**: np.arange(dtype=datetime64\['D'\]))
np.arange('2016-07','2016-08',dtype='datetime64[D]')


#### 35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆) 

(**hint**: np.add(out=), np.negative(out=), np.multiply(out=), np.divide(out=))

a=np.random.randint(1,10,3);a
b=np.random.randint(1,10,3);b
np.multiply((np.add(a,b)),np.divide(np.negative(a),2))


A = np.ones(3)*1;A
B = np.ones(3)*2;B
C = np.ones(3)*3;C
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)




#### 36. Extract the integer part of a random array using 5 different methods (★★☆) 

(**hint**: %, np.floor, np.ceil, astype, np.trunc)
a=np.random.rand(5)+1;a
a-a%1
np.floor(a)
np.ceil(a)-1
a.astype(int)
np.trunc(a)


Z = np.random.uniform(0,10,10);Z
print (Z - Z%1)
print (np.floor(Z))
print (np.ceil(Z)-1)
print (Z.astype(int))
print (np.trunc(Z))

#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆) 

(**hint**: np.arange)
np.random.randint(0,4,(5,5))
np.arange(25).reshape(5,5)

Z = np.zeros((5,5))
Z+=np.arange(5)
print(Z)

#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆) 

(**hint**: np.fromiter)
np.random.randint(0,10,10)
np.random.randn(2,3,10)

def generate():
    for x in range(10):
        yield x
Z = np.fromiter(generate(),dtype=float,count=-1)
print(Z)

#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆) 

(**hint**: np.linspace)
np.linspace(0,1,10)

np.linspace(0,1,12,endpoint=True)[1:-1]

#### 40. Create a random vector of size 10 and sort it (★★☆) 

(**hint**: sort)
a=np.random.random(10);a
np.sort(a)
a.sort();a
#### 41. How to sum a small array faster than np.sum? (★★☆) 

(**hint**: np.add.reduce)
x=np.arange(1,10);x
np.add.reduce(x)

np.multiply.reduce(x)
#### 42. Consider two random array A and B, check if they are equal (★★☆) 

(**hint**: np.allclose, np.array\_equal) 
A=np.random.randint(0,10,10);A
B=np.random.randint(0,10,10);B
np.allclose(A,B)
np.array_equal(A,B)

??#### 43. Make an array immutable (read-only) (★★☆) 

(**hint**: flags.writeable)

Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1

??#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆) 

(**hint**: np.sqrt, np.arctan2)
np.random.random((10,2))
Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)

#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆) 

(**hint**: argmax
a=np.random.random(10);a
a[np.argmax(a)]=0
np.argmax(a)
a.max()
a
#### 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆) 

(**hint**: np.meshgrid)
Z=np.zeros((5,5),[('x',float),('y',float)])
Z['x'],Z['y']=np.meshgrid(np.linspace(0,1,5),np.linspace(0,1,5))
Z

####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) 

#####  (hint: np.subtract.outer)
x=np.arange(8)
y=x+0.5
c=1.0/np.subtract.outer(x,y)
print(np.linalg.det(c))

#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆) 

(**hint**: np.iinfo, np.finfo, eps)
for i in [np.int8,np.int32,np.int64]:
    print(np.iinfo(i).min)
    print(np.iinfo(i).max)
for i in [np.float32,np.float64]:
    print(np.finfo(i).min)
    print(np.finfo(i).max)
    print(np.finfo(i).eps)


#### 49. How to print all the values of an array? (★★☆) 

(**hint**: np.set\_printoptions)
z=np.random.randint(0,10,(5,5));z
np.set_printoptions(z)
z

np.set_printoptions(threshold=np.nan)
Z = np.zeros((16,16))
print(Z)

#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆) 

(**hint**: argmin)
Z = np.arange(100);Z
v = np.random.uniform(0,100);v
index = (np.abs(Z-v)).argmin()
print(Z[index])


#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆) 

(**hint**: dtype)
Z=np.zeros(10,[('position',[('x', float, 1),('y', float, 1)]), ('color',[('r', float, 1),('g', float, 1),('b', float, 1)])])
print(Z)


#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆) 

(**hint**: np.atleast\_2d, T, np.sqrt)

Z=np.random.random((10,2));Z
X,Y=np.atleast_2d(Z[:,0], Z[:,1]);X,Y
D=np.sqrt((X-X.T)**2+(Y-Y.T)**2);D


#--------------------------------------
# 使用scipy更快
import scipy.spatial
Z = np.random.random((10,2))
D = scipy.spatial.distance.cdist(Z,Z)
print(D)

#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place? 

(**hint**: view and [:] = )
x=np.arange(10,dtype=np.float32);x
x=x.astype(np.int32,copy=False)
x
#### 54. How to read the following file? (★★☆) 

(**hint**: np.genfromtxt)


```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```

#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆) 

(**hint**: np.ndenumerate, np.ndindex)
a=np.arange(20).reshape(4,5)
for index,value in np.ndenumerate(a):
    print(index,value)

for index in np.ndindex(a.shape):
    print(index,a[index])


#### 56. Generate a generic 2D Gaussian-like array (★★☆) 

(**hint**: np.meshgrid, np.exp)


#### 57. How to randomly place p elements in a 2D array? (★★☆) 

(**hint**: np.put, np.random.choice)
n=5
p=6
Z=np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
print(Z)

a=np.arange(5);a
np.put(a,[0,2],[-44,-55]);a
np.random.choice(5,3)
np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
np.random.choice(5, 3, replace=False)
np.random.permutation(np.arange(5))[:3]
#### 58. Subtract the mean of each row of a matrix (★★☆) 

(**hint**: mean(axis=,keepdims=))
x=np.random.rand(5,10)
x.mean()
x-x.mean(axis=0,keepdims=True)

x-x.mean(axis=1).reshape(-1, 1)
#### 59. How to sort an array by the nth column? (★★☆) 

(**hint**: argsort)
x=np.random.randint(0,10,(5,5))
x[x[:,2].argsort()]


#### 60. How to tell if a given 2D array has null columns? (★★☆) 

(**hint**: any, ~)
x=np.random.randint(1,10,(3,10))
x=np.zeros((5,5));x
(~x.any(axis=0)).any()


#### 61. Find the nearest value from a given value in an array (★★☆) 

(**hint**: np.abs, argmin, flat)
Z=np.random.uniform(0,1,10);Z
z=0.6
m=Z.flat[np.abs(Z - z).argmin()];m


#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆) 

(**hint**: np.nditer)



#### 63. Create an array class that has a name attribute (★★☆) 

(**hint**: class method)



#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★) 

(**hint**: np.bincount | np.add.at)



#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★) 

(**hint**: np.bincount)



#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★) 

(**hint**: np.unique)



#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★) 

(**hint**: sum(axis=(-2,-1)))



#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★) 

(**hint**: np.bincount)



#### 69. How to get the diagonal of a dot product? (★★★) 

(**hint**: np.diag)



#### 70. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★) 

(**hint**: array\[::4\])



#### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★) 

(**hint**: array\[:, :, None\])
A=np.ones((5,5,3));A
B=2*np.ones((5,5));B
print(A*B[:,:,None])


#### 72. How to swap two rows of an array? (★★★) 

(**hint**: array\[\[\]\] = array\[\[\]\])
a=np.arange(24).reshape(4,6);a
a[[0,2]] = a[[2,0]];a



#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★) 

(**hint**: repeat, np.roll, np.sort, view, np.unique)



#### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★) 

(**hint**: np.repeat)



#### 75. How to compute averages using a sliding window over an array? (★★★) 

(**hint**: np.cumsum)



#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is  shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]) (★★★) 

(**hint**: from numpy.lib import stride\_tricks)



#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★) 

(**hint**: np.logical_not, np.negative)



#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\[i\],P1\[i\])? (★★★)



#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\[j\]) to each line i (P0\[i\],P1\[i\])? (★★★)



#### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★) 

(**hint**: minimum, maximum)



#### 81. Consider an array Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]? (★★★) 

(**hint**: stride\_tricks.as\_strided)



#### 82. Compute a matrix rank (★★★) 

(**hint**: np.linalg.svd)



#### 83. How to find the most frequent value in an array? 

(**hint**: np.bincount, argmax)



#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★) 

(**hint**: stride\_tricks.as\_strided)



#### 85. Create a 2D array subclass such that Z\[i,j\] == Z\[j,i\] (★★★) 

(**hint**: class method)



#### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★) 

(**hint**: np.tensordot)



#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★) 

(**hint**: np.add.reduceat)



#### 88. How to implement the Game of Life using numpy arrays? (★★★)



#### 89. How to get the n largest values of an array (★★★) 

(**hint**: np.argsort | np.argpartition)



#### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★) 

(**hint**: np.indices)



#### 91. How to create a record array from a regular array? (★★★) 

(**hint**: np.core.records.fromarrays)



#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★) 

(**hint**: np.power, \*, np.einsum)



#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★) 

(**hint**: np.where)



#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. \[2,2,3\]) (★★★)



#### 95. Convert a vector of ints into a matrix binary representation (★★★) 

(**hint**: np.unpackbits)



#### 96. Given a two dimensional array, how to extract unique rows? (★★★) 

(**hint**: np.ascontiguousarray | np.unique)



#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★) 

(**hint**: np.einsum)



#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)? 

(**hint**: np.cumsum, np.interp)



#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★) 

(**hint**: np.logical\_and.reduce, np.mod)



#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★) 

(**hint**: np.percentile)

