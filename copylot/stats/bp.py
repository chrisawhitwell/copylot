import numpy as np
from .runs_test import runs

def bp1966(u_primes,window_length):
    u_ms = np.empty(n_windows)
    n_windows = len(u_prime)/window_size

    ind = 0
    for i in range(n_windows):
        u_ms[i] = np.mean(np.square(u_prime[ind:ind+window_size]))
        ind += window_size
    
    stationary,z = runs(u_ms)

    return stationary,z

