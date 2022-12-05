import numpy as np
from scipy import stats

def runs(x,axis=0,window_size=None,warnings=True,**kwargs):
    if not window_size:
        x_ms = x
    else:
        n_windows = int(np.ceil(len(x)/window_size))
        x_ms = np.empty(n_windows)

        ind = 0
        for i in range(n_windows):
            x_ms[i] = np.mean(np.square(x[ind:ind+window_size]))
            ind += window_size
    if warnings == True:
        if len(x_ms) < 30: print(f'Warning: Sample size ({len(x_ms)}) is less than 30. Central Limit Theorem may not hold')
    
    z = test(x_ms,**kwargs)

    return z

def test(x,axis=0,v=None,alpha=0.05,method='mean'):
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
    print(x.shape)
    if v == None:
        if method == 'mean':
            x_m = np.nanmean(x,axis=axis)    
        elif method == 'median':
            x_m = np.nanmedian(x,axis=axis)
        else:
            raise ValueError
    else:
        x_m = v

    x_p = np.roll(x,1,axis=axis)

    runs = np.nansum((x>=x_m)*(x_p<x_m) + (x < x_m)*(x_p >= x_m),dtype='int64',axis=axis)
    n1 = np.nansum(x>= x_m,dtype='int64',axis=axis)
    n2 = len(x)-n1
    print(n1)
    print(n2)
    runs_exp = ((2*n1*n2)/(n1+n2))+1
    stan_dev = np.sqrt((2*n1*n2*(2*n1*n2-n1-n2))/(((n1+n2)**2)*(n1+n2-1)))
    print(stan_dev)
    z = (runs-runs_exp)/stan_dev 

    stationary = abs(z) < stats.norm.ppf(1-alpha/2)

    return stationary,z