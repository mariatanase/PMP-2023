
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(12.5, 3.5))
traffic_data = np.genfromtxt("C:/Users/Maria/Downloads/trafic.csv", delimiter=',')
x_values = traffic_data[:, 0]
y_values = traffic_data[:, 1]
n_traffic_data = len(traffic_data)
plt.bar(x_values, y_values, color="#348ABD")
plt.xlabel("Minute")
plt.ylabel("Masini")
plt.title("Trafic")
plt.xlim(0, n_traffic_data)

#plt.show()

with pm.Model() as model:
    alpha = 1.0/traffic_data[:, 1].mean()

    parameter_1 = pm.Exponential("poisson_param1", alpha)
    parameter_2 = parameter_1 + pm.Exponential("poisson_param2", alpha)
    parameter_3 = parameter_2 - pm.Exponential("poisson_param3", alpha)
    parameter_4 = parameter_3 + pm.Exponential("poisson_param4", alpha)
    parameter_5 = parameter_4 - pm.Exponential("poisson_param5", alpha)

    time = pm.DiscreteUniform("time", lower=1, upper=n_traffic_data)
    lambda_value = pm.math.switch(time < 7 * 60, parameter_1, 
                                  pm.math.switch(time < 8 * 60, parameter_2, 
                                                 pm.math.switch(time < 16 * 60, parameter_3, 
                                                                pm.math.switch(time < 19 * 60, parameter_4, parameter_5))))
    
    observation = pm.Poisson("obs", lambda_value, observed=traffic_data)

    trace = pm.sample(10000, tune=5000)

    pm.summary(trace)

