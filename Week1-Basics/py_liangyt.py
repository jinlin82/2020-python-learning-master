## 网盘视频 代码

3+2
import matplotlib.pyplot
import math 

math.sin(3.14)

import numpy

numpy.random.rand(10)

import numpy as np 
np.sinc(0)

import matplotlib.pyplot
matplotlib.pyplot.plot([1,10])

import matplotlib.pyplot as plt 
plt.plot([1,10])

import math as m 
m.sin(0)

from math import sin 
sin(0)

a=[1,2,3]
import numpy as np
a=np.array(a)
a+1

## py_basis_beamer 代码

# if Statement example
x = int(input("Please enter an integer: ")) 
if x < 0:
    x = 0
    print('Negative changed to zero')
elif x == 0:
    print('Zero')
elif x == 1:
    print('Single')
else:
    print('More')

# for Statements
words = ["cat",'window','defenestrate']
for w in words:
    print(w,len(w))

# Defining Functions Examples ???
def fib(n): 
    a, b = 0, 1
    while a < n:
          print a,
          a, b = b, a+b

# set examples
basket = ['apple','orange','apple','pear','orange','banana']
fruit = set(basket)
fruit
set(['orange','pear','apple','banana'])
'orange' in fruit
'crabgrass' in fruit 

# Demonstrate set operations on unique letters from two words
a = set('abracadabra')
b = set('alacazam')
a
a - b
a | b  #并集
a & b  #交集
a ^ b  # letters in a or b but not both

# dictionary examples
tel = {'jack': 4098, 'sape': 4139}
tel['guido'] = 4127
tel
 tel['jack']

del tel['sape']  #删除：del
tel['irv'] = 4127
tel 
tel.keys()
'guido' in tel

# 索引与切片
x = list(range(10)) #取0~9十个数字
x[2]
x[1:5]  #左关右开
x[-1]   #负整数索引是从尾部开始取 
x[2:-2]

# 下标值超出数组长度范围，不会造成越界错误
x[0:11]
x[0:12]
x[0:]
x[:]

# 倒序取元素,返回一个空的数组
x[3:1]
x[-1:1]

#  x[a:b:step]
x[1:7:2] # 如果 a 在 b 前面，step 要取正数
x[1:7:-2] 
x[7:1:2] 
x[7:1:-2] #如果 a 在 b 后面，step 要取负数
x[::-1] 
x[::1]

# Module import
import math
math.sin(3)

# 查看某一 package 中所有子包和子模块 ???
import pkgutil
import numpy
for importer, modname, ispkg in pkgutil.iter_modules(numpy.__ path__, prefix="numpy."):
    print(modname) 

# ???
class Point:
     """Represents a point in 2-D space.""" 
Point()

# ???
blank.x = 3.0
blank.y = 4.0
blank.y
x = blank.x
x 

#methods
a=[1,2,3]
import numpy as np
a=np.array(a)
a+1

