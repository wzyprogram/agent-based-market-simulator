'''
Queueing(one-side LOB)
Concept, not real code
'''

init X = [0]*K; t=0
precompute lambdas = [lambda0 * np.exp(-k*(i)) for i in range(K)]  # or power-law

def best_ask(X):
    for i,x in enumerate(X):
        if x>0: return i
    return None  # empty book

while t < T_end:
    arrival = sum(lambdas)
    cancel  = theta * sum(X)
    rateTot = arrival + mu + cancel
    dt = np.random.exponential(1.0/rateTot)
    t += dt

    r = np.random.rand() * rateTot
    # 1) limit arrival
    if r < arrival:
        # choose price level i with prob ~ lambdas[i]
        i = np.searchsorted(np.cumsum(lambdas), np.random.rand()*arrival)
        X[i] += 1
    else:
        r -= arrival
        # 2) market buy
        if r < mu:
            i = best_ask(X)
            if i is not None: X[i] -= 1
        else:
            r -= mu
            # 3) cancellation: choose level proportional to X[i]
            cum = np.cumsum([theta*xi for xi in X])
            i = np.searchsorted(cum, r)
            if X[i] > 0: X[i] -= 1

    # logging every k steps
    # record X copy, best_ask, cumulative depth curve, etc.

