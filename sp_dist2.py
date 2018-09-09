import numpy as np


"""
DIST2	Calculates squared distance between two sets of points.
"""

def sp_dist2(x, c):

   ndata, dimx = np.shape(x)
   ncentres, dimc = np.shape(c)

   if dimx != dimc:
       raise ValueError("Data dimension does not match dimension of centres")

   first_a = np.sum((x ** 2).T, 0)
   first_a = np.reshape(first_a, (1, np.shape(first_a)[0]))
   a = np.dot(np.ones((ncentres, 1)), first_a).T
   second_b = np.sum((c ** 2).T, 0)
   second_b = np.reshape(second_b, (1, np.shape(second_b)[0]))
   b = np.dot(np.ones((ndata, 1)), second_b)
   d = np.dot(x, c.T) * 2
   n2 = a + b - d

   # Rounding errors occasionally cause negative entries in n2
   for i in range(np.shape(n2)[0]):
       for j in range(np.shape(n2)[1]):
           if n2[i, j] < 0:
               n2[i, j] = 0

   return n2

