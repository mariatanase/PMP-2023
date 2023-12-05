import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

data = pd.read_csv("C:/Users/Maria/Downloads/Admission.csv")

y_1 = data['Admission'].values
x_n = ['GRE', 'GPA']
x_1 = data[x_n].values

if __name__ == '__main__':
    #1
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10) #beta0
        beta = pm.Normal('beta', mu=0, sigma=2, shape=len(x_n))
        mu = alpha + pm.math.dot(x_1, beta)
        theta = pm.Deterministic('theta', pm.math.sigmoid(mu))
        bd = pm.Deterministic('bd', -alpha / beta[1] - beta[0] / beta[1] * x_1[:, 0])
        y1 = pm.Bernoulli('y1', p=theta, observed=y_1)
        idata = pm.sample(2000, cores=1, return_inferencedata=True)
    
    idx = np.argsort(x_1[:,0])
    bd = idata.posterior['bd'].mean(("chain", "draw"))[idx]
    plt.scatter(x_1[:,0], x_1[:,1], c=[f'C{x}' for x in y_1])
    plt.plot(x_1[:,0][idx], bd, color='k')
    az.plot_hdi(x_1[:,0], idata.posterior['bd'], color='k')
    plt.xlabel(x_n[0])
    plt.ylabel(x_n[1])
    plt.show()