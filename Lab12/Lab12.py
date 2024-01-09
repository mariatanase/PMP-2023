import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def posterior_grid(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    prior = abs(grid - 0.5)
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

data = np.repeat([0, 1], (30, 9))
points = 10
h = data.sum()
t = len(data) - h
grid, posterior = posterior_grid(points, h, t)
plt.plot(grid, posterior, 'o-')
plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ')
plt.show()

def estimate_pi_errors(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum()*4/N
    error = abs((pi - np.pi) / pi) * 100
    return error

Ns = [100, 1000, 10000]

for N in Ns:
    errors = [estimate_pi_errors(N) for _ in range(100)]

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    print(mean_error)
    print(std_error)

    plt.errorbar(N, mean_error, yerr=std_error, fmt='o-', label=f'N = {N}', capsize=5)
    plt.show()

def metropolis(func, draws=10000):
    trace = np.zeros(draws)
    old_x = 0.5 # func.mean()
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace

n_trials = [0, 1, 2, 3, 4, 8, 16, 32, 50, 150]
data = [0, 1, 1, 1, 1, 4, 6, 9, 13, 48]
theta_real = 0.35
beta_params = [(1, 1), (20, 20), (1, 4)]
for (a, b) in beta_params:
    func = stats.beta(a, b)
    trace = metropolis(func=func)
    x = np.linspace(0.01, .99, 100)
    y = func.pdf(x)
    plt.xlim(0, 1)
    plt.plot(x, y, 'C1-', lw=3, label='True distribution')
    plt.hist(trace[trace > 0], bins=25, density=True, label='Estimated distribution')
    plt.xlabel('x')
    plt.ylabel('pdf(x)')
    plt.yticks([])
    plt.legend()
    plt.show()

