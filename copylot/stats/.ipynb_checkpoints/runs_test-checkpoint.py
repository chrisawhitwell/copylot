import numpy as np
from scipy import stats

def runs(x,window_size=None,**kwargs):
    if not window_size:
        x_ms = x
    else:
        n_windows = int(len(x)/window_size)
        x_ms = np.empty(n_windows)

        ind = 0
        for i in range(n_windows):
            x_ms[i] = np.mean(np.square(x[ind:ind+window_size]))
            ind += window_size
    
    stationary,z = test(x_ms,**kwargs)

    return stationary,z

def test(x,v=None,alpha=0.05,method='mean'):
    """
    Runs test for randomness.

    Parameters
    ----------
    x : np.array
        Sequence to test
    v : int or None
        Value about which to test for runs, if None - will use 'method' to determine
    alpha :  int
        Confidence level for two-tailed distribution

    Returns
    ---------
    stationary: bool
        True is stationary, otherwise False

    Assumes normal distribution
    """
    if v == None:
        if method == 'mean':
            x_m = np.nanmean(x)    
        elif method == 'median':
            x_m = np.nanmedian(x)
        else:
            raise ValueError
    else:
        x_m = v

    x_p = np.roll(x,1)

    runs = np.sum((x>=x_m)*(x_p<x_m) + (x < x_m)*(x_p >= x_m),dtype='int64')
    n1 = np.sum(x>= x_m,dtype='int64')
    n2 = len(x)-n1

    runs_exp = ((2*n1*n2)/(n1+n2))+1
    stan_dev = np.sqrt((2*n1*n2*(2*n1*n2-n1-n2))/(((n1+n2)**2)*(n1+n2-1)))
    z = (runs-runs_exp)/stan_dev 

    stationary = abs(z) < stats.norm.ppf(1-alpha/2)

    return stationary,z