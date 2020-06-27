3+2

import matplotlib.pyplot
matplotlib.pyplot.plot([1,10])

import math
math.sin(3.14)
math.cos(0)

import numpy

numpy.random.rand(10)
import numpy as np
np.sinc(0)

import matplotlib.pyplot as plt
plt.plot([2,5])

import math as m
m.sin(0)
from math import sin
sin(0)

import numpy.compat
import matplotlib.pyplot

a=[1,2,3]
a
import numpy as np
a=np.array(a)
a+1
a=np.sum(a)
a+2
np.sin(a)


x=int(input("please enter an integer:"))
if x<0:
    x=0
    print('Negative changed to zero')
elif x==0:
    print('Zero')
elif x==1:
    print('Single')
else:
    print('More')


words=["cat","window","defenestrate"]
for w in words:
  print(w,len(w))


range(10)
range(1,10)

def fib(n):
  a,b=0,1
  while a<n:
    print (a),
    a,b = b,a+b


basket=['apple','orange','apple','pear','orange','banbana']
fruit=set(basket)
fruit
"orange" in fruit
"watermelon" in fruit

a=set('abracadabra')
b=set('alacazanm')
a
b
a-b
a|b
a&b
a^b

tel={'jack':4098,'sape':4139}
tel['guido']=4127
tel
tel['jack']
tel['irv']=4127
tel
tel.keys()
tel.values()
'guido' in tel
'abc' in tel
tel
del tel['sape']
tel

x=list(range(10))
x[1:10]
x[2]
x[1:5]
x[-1]
x[2:-2]

x=list(range(20))
x[0:15]
x[0:22]
x[0:]
x[:]

x[3:1]
x[-1:1]
x[1:-1]

x[1:7:2]
x[1:7:-2]
x[7:1:-2]
x[7:1:2]
x[::-1]
x[::3]
x[::-2]

import math
math.sin(3)

import pkgutil
import numpy

for importer, modname, ispkg in pkgutil.iter_modules(numpy._path_,  prefix="numpy."):
  print(modname)


class Point:
    """Represents a point in 2-D space."""
Point()
""""""

Point
blank=Point()
blank

blank.x=3.0
blank.y=4.0
blank.y
x=blank.x
x