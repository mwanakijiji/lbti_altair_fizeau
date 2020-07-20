## some code to make discrete CDFs
from bisect import bisect_left
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import seaborn as sns

def discrete_cdf(input_array):
    
    input_array_sorted = np.sort(input_array)
    len_array = len(input_array_sorted)        

    return input_array_sorted, np.cumsum(input_array_sorted)/np.max(np.cumsum(input_array_sorted))

x = np.random.randn(10000) # generate samples from normal distribution (discrete data)
norm_cdf = scipy.stats.norm.cdf(x) # calculate the cdf - also discrete

print(x)

cdf = discrete_cdf(x)
xvalues = np.sort(x)
yvalues = [cdf(point) for point in xvalues]
plt.plot(xvalues, yvalues, label="mine")


# plot the cdf
#sns.lineplot(x=x, y=norm_cdf)

plt.close()
plt.plot(x,norm_cdf,label="theirs")

ans = discrete_cdf(x)

#iplt.plot(ans[0],ans[1],label="mine")
plt.legend()

plt.show()

import ipdb; ipdb.set_trace()
