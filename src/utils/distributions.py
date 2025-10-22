import numpy as np

def power_law(xm, alpha):
    """Draw volume from power-law"""
    return xm / (np.random.rand() ** (1 / alpha))

def rexp(theta, theta_min, theta_max):
    """Truncated exponential"""
    while True:
        x = np.random.exponential(theta)
        if theta_min <= x <= theta_max:
            return x

def agent_times(theta, theta_min, theta_max, T):
    """Generate action times"""
    times = [0.0]
    counter = 0.0
    while counter < T:
        tau = rexp(theta, theta_min, theta_max) * 1000
        counter += tau
        times.append(counter)
    return times