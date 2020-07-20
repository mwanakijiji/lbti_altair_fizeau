import numpy as np
import matplotlib.pyplot as plt

N = 100
Z = np.random.normal(size = N)
# method 1
H,X1 = np.histogram( Z, bins = 10, normed = True )
dx = X1[1] - X1[0]
F1 = np.cumsum(H)*dx
#method 2
X2 = np.sort(Z)
F2 = np.array(range(N))/float(N)

plt.plot(X1[1:], F1)
plt.plot(X2, F2)
plt.show()
