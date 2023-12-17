import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

az.style.use('arviz-darkgrid')
dummy_data = np.loadtxt('C:/Users/Maria/Downloads/dummy.csv')


if __name__ == '__main__':
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]

    order = 5
    x_1p = np.vstack([x_1**i for i
    in range(1, order+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))
    x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


    #1a
    with pm.Model() as model_p1:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=10, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
        idata_p = pm.sample(1000, cores=1, return_inferencedata=True)
    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()

    #1b
    with pm.Model() as model_p2:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=100, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
        idata_p1 = pm.sample(1000, cores=1, return_inferencedata=True)
    α_p_post = idata_p1.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p1.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()

    with pm.Model() as model_p3:
        α = pm.Normal('α', mu=0, sigma=1)
        sd=np.array([10, 0.1, 0.1, 0.1, 0.1])
        β = pm.Normal('β', mu=0, sigma=sd, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
        idata_p2 = pm.sample(1000, cores=1, return_inferencedata=True)
    α_p_post = idata_p2.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p2.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()

    #2
    x_1 = np.random.normal(-10, 10, 500)
    y_1 = np.random.normal(-10, 10, 500)

    order = 5
    x_1p = np.vstack([x_1**i for i
    in range(1, order+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))
    x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


    with pm.Model() as model_p1:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=10, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
        idata_p = pm.sample(1000, cores=1, return_inferencedata=True)
    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()


    with pm.Model() as model_p2:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=100, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
        idata_p1 = pm.sample(1000, cores=1, return_inferencedata=True)
    α_p_post = idata_p1.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p1.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()


    with pm.Model() as model_p3:
        α = pm.Normal('α', mu=0, sigma=1)
        sd=np.array([10, 0.1, 0.1, 0.1, 0.1])
        β = pm.Normal('β', mu=0, sigma=sd, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
        idata_p2 = pm.sample(1000, cores=1, return_inferencedata=True)
    α_p_post = idata_p2.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p2.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()

    #3
    x_1 = np.random.normal(-10, 10, 500)
    y_1 = np.random.normal(-10, 10, 500)

    order = 3
    x_1p = np.vstack([x_1**i for i
    in range(1, order+1)])
    x_1s_cubic = (x_1p - x_1p.mean(axis=1, keepdims=True))
    x_1p.std(axis=1, keepdims=True)
    y_1s_cubic = (y_1 - y_1.mean()) / y_1.std()
    with pm.Model() as model_cubic:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=10, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s_cubic)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s_cubic)
        idata_cubic = pm.sample(1000, cores=1, return_inferencedata=True)
    α_p_post = idata_cubic.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_cubic.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s_cubic[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s_cubic)
    plt.plot(x_1s_cubic[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s_cubic[0], y_1s_cubic, c='C0', marker='.')
    plt.legend()
    plt.show()


    order = 2
    x_1p = np.vstack([x_1**i for i
    in range(1, order+1)])
    x_1s_patratic = (x_1p - x_1p.mean(axis=1, keepdims=True))
    x_1p.std(axis=1, keepdims=True)
    y_1s_patratic = (y_1 - y_1.mean()) / y_1.std()
    with pm.Model() as model_patratic:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=10, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s_patratic)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s_patratic)
        idata_patratic = pm.sample(1000, cores=1, return_inferencedata=True)

    order = 1
    x_1p = np.vstack([x_1**i for i
    in range(1, order+1)])
    x_1s_liniar = (x_1p - x_1p.mean(axis=1, keepdims=True))
    x_1p.std(axis=1, keepdims=True)
    y_1s_liniar = (y_1 - y_1.mean()) / y_1.std()
    with pm.Model() as model_liniar:
        α = pm.Normal('α', mu=0, sigma=1)
        β = pm.Normal('β', mu=0, sigma=10, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s_liniar)
        y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s_liniar)
        idata_liniar = pm.sample(1000, cores=1, return_inferencedata=True)

    pm.compute_log_likelihood(idata_cubic,model=model_cubic)
    cubic_waic = az.waic(idata_cubic)
    pm.compute_log_likelihood(idata_patratic,model=model_patratic)
    patratic_waic = az.waic(idata_patratic)
    pm.compute_log_likelihood(idata_liniar,model=model_liniar)
    liniar_waic = az.waic(idata_liniar)

    cubic_loo = az.loo(idata_cubic)
    patratic_loo = az.loo(idata_patratic)
    liniar_loo = az.loo(idata_liniar)

    print("WAIC - Cubic Model:", cubic_waic)
    print("WAIC - Quadratic Model:", patratic_waic)
    print("WAIC - Linear Model:", liniar_waic)

    print("\nLOO - Cubic Model:", cubic_loo)
    print("LOO - Quadratic Model:", patratic_loo)
    print("LOO - Linear Model:", liniar_loo)

    comparison_result = az.compare({'Cubic Model': idata_cubic, 'Quadratic Model': idata_patratic, 'Linear Model': idata_liniar}, method='BB-pseudo-BMA', ic='waic', scale='deviance')
    print(comparison_result)
    az.plot_compare(comparison_result)
    plt.show()

    comparison_result = az.compare({'Cubic Model': idata_cubic, 'Quadratic Model': idata_patratic, 'Linear Model': idata_liniar}, method='BB-pseudo-BMA', ic='loo', scale='deviance')
    print(comparison_result)
    az.plot_compare(comparison_result)
    plt.show()