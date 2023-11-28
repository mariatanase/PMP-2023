import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import numpy as np

data = pd.read_csv("C:/Users/Maria/Downloads/Prices.csv")

#1
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', 5)
    mu = alpha + beta1 * data['Speed'] + beta2 * np.log(data['HardDrive'])
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=data['Price'])
    idata = pm.sample(2000, tune=1000, cores=1, return_inferencedata=True)

az.plot_posterior(idata, var_names=['alpha', 'beta1', 'beta2', 'sigma'])
plt.show()
'''pm.sample_posterior_predictive(idata, model=model, random_seed=2, extend_inferencedata=True)
ax = az.plot_ppc(idata, num_pp_samples=200, figsize=(12, 6), mean=True)
plt.xlim(0, 12)
plt.show()'''

#2
az.plot_forest(idata, var_names=['beta1', 'beta2'], combined=True, hdi_prob=0.95)
plt.show()
summary_data = az.summary(idata, var_names=['beta1', 'beta2'], hdi_prob=0.95)
mean_beta1 = summary_data.loc['beta1', 'mean']
mean_beta2 = summary_data.loc['beta2', 'mean']
print(f"Estimare pentru beta1: {mean_beta1}")
print(f"Estimare pentru beta2: {mean_beta2}")

#3
#Observam atat in graficul generat, cat si in rezumat ca valorile estimate (mean) ale variabilelor beta1 si beta2 sunt 
#diferite de 0. Acest lucru ne da de inteles ca atat frecventa procesorului, cat si marimea hard diskului au o anumita 
#influenta asupra pretului de vanzare al PC-urilor. Totusi, valoarea estimata a lui beta2 (ex. 266.988) mai mare decat cea
#a lui beta1 (ex. 11.250) semnifica faptul ca marimea hard diskului este un predictor cu mult mai util al pretului de vanzare.

#Bonus
#Introducen variabila Premium in model
data['Premium_Binary'] = (data['Premium'] == 'yes').astype(int)
with pm.Model() as model_premium:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    beta3 = pm.Normal('beta3', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', 5)
    mu = alpha + beta1 * data['Speed'] + beta2 * np.log(data['HardDrive']) + beta3 * data['Premium_Binary']
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=data['Price'])
    idata = pm.sample(2000, tune=1000, cores=1, return_inferencedata=True)
az.plot_forest(idata, var_names=['beta1', 'beta2', 'beta3'], combined=True, hdi_prob=0.95)
plt.show()
summary_data = az.summary(idata, var_names=['beta3'], hdi_prob=0.95)
mean_beta3 = summary_data.loc['beta3', 'mean']
hdi_beta3 = summary_data.loc['beta3', 'hdi_2.5%'], summary_data.loc['beta3', 'hdi_97.5%']
print(f"Estimare pentru beta3: {mean_beta3}")
print(f"Interval de credibilitate 95% pentru beta3: ({hdi_beta3[0]}, {hdi_beta3[1]})")
#Intervalul de credibilitate pentru 'beta3' este destul de larg si include valoarea zero.
#Asa ca nu exista destul dovezi pentru a sustine ca efectul variabilei este diferit de zero.
#Inseamna ca variabila 'Premium' nu este foarte semnificativa in ceea ce priveste predictia pretului.