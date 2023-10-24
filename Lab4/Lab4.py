import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

lambda_Poisson = 20
mean_plasare = 2
standard_deviation_plasare = 0.5
alpha = 4
r = []
total = []

for i in range(10):
    n = np.random.poisson(lambda_Poisson)
    timp_plasare = np.random.normal(mean_plasare, standard_deviation_plasare)
    timp_gatire = np.random.exponential(alpha)
    timp_total = timp_gatire + timp_plasare
    r.append((n, timp_plasare, timp_gatire))
    total.append(timp_total)
print(r)
print(total)
