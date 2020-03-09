
import numpy as np
 
a=np.array([1, 2, 3, 4, 5, 6, 7, 8])
print (a)
f=a.reshape(2, 2, 2)
print (f)
print(f.reshape(-1,1).shape)

x = lambda t:t+1
print(x(1))