import numpy as np
a=np.arange(4)
b=np.array([2,5,8,9])
a*b

A=np.arange(12).reshape(3,4)
B=np.arange(13,25).reshape(4,3)
np.dot(A, B)
A.dot(B)
A.sum()
A.sum(axis=0)
A.sum(axis=1)

"""END"""

A=np.arange(12).reshape(3,4)
np.exp(A)
np.sqrt(A)

"""END"""

x=np.arange(12)**2
x[3]
x[2:6]
x[7:]
x[::-1]
x[9:2:-3]

"""END"""

A=np.arange(24).reshape(4,6)
A[2,3]
A[1:3, 2:4]
A[1]
A[:, 2:4]
A[..., 3]

"""END"""

import numpy as np
A=np.arange(24).reshape(4,6)
for i in A:
    """打印A的各行"""
    print(i) 

for i in A.flat:
    """打印A中的每个元素"""
    print(i)

"""END"""

import numpy as np
a = np.floor(10 * np.random.random((3,4)))
a.shape



a.ravel()
a.T
a.reshape(2,6)
a.resize(2,6)

"""END"""

a = np.arange(12) ** 2 # the first 12 square numbers
i = np.array( [ 1,1,3,8,5 ] ) # an array of indices
a[i] # the elements of a at the positions i
np.array([ 1, 1, 9, 64, 25])

j = np.array( [ [ 3, 4], [ 9, 7 ] ] )
a[j]



"""END"""

a = np.arange(12).reshape(3,4)
i = np.array([[0,1], [1,2]])
j = np.array([[2,1], [3,3]])

a[i]
a[i,j]
a[i, 2]
a[:,j]

L = [i,j]
a[L]

"""END"""

a = np.arange(12).reshape(3,4)
b=a>4
a[b]
a[b]=0

a = np.arange(12).reshape(3,4)
b1 = np.array([False,True,True])
b2 = np.array([True,False,True,False])
a[b1,:]
a[b1]
a[:,b2]
a[b1,b2]

"""END"""

x = np.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],
             dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
x['name']
x[['name', 'age']]

"""END"""

a = np.array([2,3,4,5])
b = np.array([8,5,4])
c = np.array([5,4,6,8,3])
ax,bx,cx = np.ix_(a,b,c)

result = ax+bx * cx
result

"""END"""
