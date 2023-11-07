import pymc as pm
import arviz as az
import numpy as np

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

results = {}

for Y in Y_values:
    for theta in theta_values:
        with pm.Model() as model:
            n = pm.Poisson("n", mu=10)
            Y_observed = pm.Binomial("Y_observed", n=n, p=theta, observed=Y)
            trace = pm.sample(2000, tune=1000, cores=1)
        results[(Y, theta)] = trace


for (Y, theta), trace in results.items():
    print(f"Y={Y}, θ={theta}:")
    az.plot_posterior(trace)