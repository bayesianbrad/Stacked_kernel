# To genereate different types of training data that is very rough and smooth use Weierstrass-M function
# Simply vary a,b 
import numpy as np
import math
from matplotlib import pyplot as plt


"""Returns toy data generated from a superposition"""
no_points = 100
x = np.linspace(0,1,no_points)
a = 0.5
b = 2.5
c = 2
d = 1.5
# does the following in math f(x)   =  \sum_{i = 0}^{n} \frac{1}{a^{i}} \cos(b^{i} * \pi * x)
F_n = lambda n: (sum([1/(pow(a,i)) * np.cos(pow(b,i) * math.pi * x) for i in range(n+1)]))
y= F_n(3) + F_n(5) 
plt.plot(x, y, label = 'F(7)+F(9)')
# for i in range(5,10):
# 	y = F_n(i)
# 	plt.plot(x, F_n(i),label = str(i))

print(y)
plt.legend(loc='upper right')
plt.show()
