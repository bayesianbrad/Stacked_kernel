# To genereate different types of training data that is very rough and smooth use Weierstrass-M function
# Simply vary a,b 
import numpy as np
import math
np.random.seed (seed = 0 )

def weierstrass(x, a=0.5, b=2.5,i=3,j=5):
   """
   Returns toy data generated from a superposition of Weierstrass M function
     Input: 
       no_points       -  int
       x               -  np.array 1 x no_points
       a,b             -  floats, representing Weierstrass function coeffiecients - see wikipeida page
       i,j             -  ints, Determine which Weierstrass functions to add together
  """
   a = 0.5
   b = 2.5
   # does the following in math f(x)   =  \sum_{i = 0}^{n} \frac{1}{a^{i}} \cos(b^{i} * \pi * x)
   F_n = lambda n: (sum([1/(pow(a,i)) * np.cos(pow(b,i) * math.pi * x) + np.random.rand(1) for i in range(n+1)]))
   y   = F_n(i) + F_n(j) 
# 3 and 5 choosen as it works well with an arbitary number of points (up to 10000 tested visually)
# for i in range(5,10):
#   y = F_n(i)
#   plt.plot(x, F_n(i),label = str(i))
# print(y)
# plt.legend(loc='upper right')
# plt.show()
   return y