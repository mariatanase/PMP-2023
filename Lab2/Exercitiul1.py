import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

m1 = stats.expon.rvs(scale=1/4) 
m2 = stats.expon.rvs(scale=1/6)

az.plot_posterior({'m1':m1,'m2':m2}) 
plt.show() 