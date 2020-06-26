##class1
3+2
3+5
#three ways to import
#way1
import matplotlib.pyplot
matplotlib.pyplot.plot([1,10])
#Question:After run the code,there is no image.


import matplotlib.pyplot as plt 
plt.plot([1,10])
plt.show() #show image

import math
math.sin(3.14)
import numpy
numpy.random.rand(10)
#general ways#
import numpy as np 
np.sin(0)

#way2
from math import sin 
sin(0)

##
import matplotlib.pyplot
import numpy.random


######cn######
import math
math.sin(3)
import math as m 
m.sin(3)

import pkgutil
import numpy
for importer,modname,ispkg in pkgutil.inter_modules(numpy._path_,prefix="numpy."):
    print(modname)

class Point:
    """Represents a point in 2-D space"""
Point()
    """"""

blank = Point()

blank.x  = 3.0
blank.y = 4.0

a=[1,2,3]
import numpy as np
a=np.array(a)
a+1 

#beamer
x=int(input("Please enter an integer:"))
if x<0:
    x=0
    print('Negative changed to zero')
elif x== 0:
    print('Zero')
elif x==1:
    print('Single')
else:
    print('More')
#Q:can not run 

words=["cat",'window','defenestrate']
for w in words:
    print(w,len(w))
""""""


basket = ['apple','orange','apple','pear','orange','banana']
fruit = set(basket)#create a set without duplication
fruit
set(['orange','pear','apple','banana'])
'orange' in fruit   #fast membership testing
'crabgrass' in fruit

#Demonstrate set operations on unique letters from two words
a = set('abracadabra')
b = set('alacazam')
a     # unique letters in a
a - b # letters in a but not in b
a | b # letters in either a or b
a & b # letters in both a and b
a ^ b # letters in a or b but not both
""""""

tel = {'jack':4098,'sape':4139}
tel['guido'] = 4127
tel
tel['jack']
del tel['sape'] # delete 'sape'
tel['irv'] = 4127
tel
{'guido':4127,'irv':4127,'jack':4098}
tel.keys()
'guido' in tel # true or false

#index（索引） and slice（切片）
x = list(range(10))
x[2]
x[1:5]
x[-1]
x[2:-2]
y = list(range(1,10))
x[0:11]
x[0:12]
x[0:]
x[:]
x[3:1]
x[-1:1]
#extended slice
x[1:7:2]
x[1:7:-2]
x[7:1:2]
x[::-1]

#module import
import math
math.sin(3)
3/2
3/2.0
3/2.00
3/1
3/1.0

a,b = 0,1
a,b = b,a+b