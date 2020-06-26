3+2

import math
math.sin(3.14)

import matplotlib.pyplot
matplotlib.pyplot.plot([1,10])

import matplotlib.pyplot as plt
plt.plot([1,10])

import math as m
m.sin(0)

from math import sin
sin(0)

dir(__builtins__)

sys.builtin_module_names

from stdlib_list import stdlib_list
libraries = stdlib_list("2.7")

help("modules")

dir()

sys.modules.keys()

a = [1,2,3]
import numpy as np
a=np.array(a)
a+1

x = int("2")
if x < 0:
    x = 0
    print('Negative changed to zero')
elif x == 0:
    print('Zero')
elif x == 1:
    print('Single')
else:
    print('More')

words = ["cat", 'window', 'defenestrate']
for w in words:
    print(w, len(w))

def fib(n):
    a, b = 0, 1
    while a < n:
        a, b = b, a+b
    print(a)

basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
fruit = set(basket)
'orange' in fruit
'crabgrass' in fruit
a = set('abracadabra')
b = set('alacazam')
a
b
a - b
a | b
a & b
a ^ b

tel = {'jack': 4098, 'sape':4139}
tel['guido'] = 4127
tel
{'sape': 4139, 'guido': 4127, 'jack': 4098}
tel['jack']
del tel['sape']
tel['irv'] = 4127
tel
tel.keys()
'guido' in tel

x = list(range(10))
x[2]
x[0:5]
x[-1]
x[2:-2]
x[0:11]
x[0:12]
x[0:]
x[:]
x[3:1]
x[-1:1]
x[1:7:2]
x[1:7:-2]
x[7:1:2]
x[7:1:-2]
x[::-1]

import pkgutil
import numpy
for impoter, modname, ispkg in pkgutil.iter_modules(numpy.__path__, prefix="numpy."):
    print(modname)

class Point():
    """Represents a point in 2-D space."""
Point()
