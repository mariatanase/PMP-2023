import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import numpy as np

df = pd.read_csv("C:/Users/Maria/Downloads/auto-mpg.csv")
df['mpg'] = pd.to_numeric(df['mpg'], errors='coerce')
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df = df.dropna(subset=['mpg', 'horsepower'])

x = df['horsepower'].values
y = df['mpg'].values

if __name__ == '__main__':
    plt.scatter(x, y)
    plt.xlabel('CP')
    plt.ylabel('mpg')
    plt.show()
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=1)
        epsilon = pm.HalfCauchy('epsilon', 5)
        mu = pm.Deterministic('mu', alpha + beta * x)
        y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y)
        idata_g = pm.sample(2000, tune=2000, return_inferencedata=True)
    plt.plot(x, y, 'C0.')
    posterior_g = idata_g.posterior.stack(samples={"chain", "draw"})
    alpha_m = posterior_g['alpha'].mean().item()
    beta_m = posterior_g['beta'].mean().item()
    plt.plot(x, alpha_m + beta_m * x, c='k', label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
    az.plot_hdi(x, posterior_g['mu'].T, hdi_prob=0.95, color='k')
    plt.xlabel('CP')
    plt.ylabel('mpg', rotation=0)
    plt.legend()
    plt.show()

#Cu cat e mai puternic motorul masinii, cu atat voi avea un consum mai mare. Astfel, voi parcurge un numar mai mic de mile per galon.
    

