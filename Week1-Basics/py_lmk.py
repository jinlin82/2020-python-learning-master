3+2

import matplotlib.pyplot
import math
math.sin(3.14)
math.cos(0)

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

import pkgutil
import numpy

for importer, modname, ispkg in pkgutil.iter_modules(numpy.__path__, prefix="numpy."):
    print(modname)

# class
class Point:
    """Represents a point in 2-D space."""
Point()

# object
>>> Point
<class '__main__.Point'>

>>> blank = Point()
>>> blank
<__main__.Point object at 0xb7e9d3ac>

# attributes
>>> blank.x = 3.0
>>> blank.y = 4.0
>>> blank.y
>>> x = blank.x
>>> x

# methods
a=[1,2,3]
import numpy as np
a=np.array(a)
a+1

# if statement
x = int(input("Please enter an integer:"))
if x < 0:
    x = 0
    print('Negative changed to zero')
elif x == 0:
    print('Zero')
elif x == 1:
    print('Single')
else:
    print('More')

# for statement
words = ["cat", 'window', 'defenestrate']
for w in words:
    print (w, len(w))

# defining functions
def fib(n):
    """Print a Fibonacci series up to n."""
    a, b = 0, 1
    while a < n:
        print a,
        a, b = b, a+b

# set
basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
fruit = set(basket)
fruit
set(['orange', 'pear', 'apple', 'banana'])
'orange' in fruit
'crabgrass' in fruit

# Demonstrate set operations on unique letters from two words
a = set('abracadabra')
b = set('alacazam')
a        # unique letters in a
a - b    # letters in a but not in b
a | b    # letters in either a or b
a & b    # letters in both a and b
a ^ b    # letters in a or b but not both

# dictionary
tel = {'jack': 4098, 'space':4139}
tel ['guido'] = 4127
tel
{'space': 4139, 'guido': 4127, 'jack': 4098}
tel['jack']
4098
del tel['space']
tel['irv'] = 4127
tel
{'guido': 4127, 'irv': 4127, 'jack':4098}
tel.keys()
['guido', 'irv', 'jack']
'guido' in tel

# index and slice
x = list(range(10))
x[2]
x[1:5]
x[-1]
x[2:-2]

x[0:11]
x[0:12]
x[0:]
x[:]

x[3:1]
x[-1:1]

# extended slice
x[1:7:2]
x[1:7:-2]
x[7:1:2]
x[7:1:-2]
x[::1]