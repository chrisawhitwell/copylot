import numpy as np

def _get_kInd(k,kmin=None,kmax=None):
    if not kmin: kmin = k[0]
    if not kmax: kmax =  k[-1]
    
    tt = np.argwhere((k <= kmax) & (k >=kmin)).flatten()
    
    while len(tt) < 9:
        if tt[0] > 0: tt = np.arange(tt[0]-1,tt[-1]+1)
        else:
            if tt[-1] == kmax:
                break
            else:
                tt = np.arange(tt[0],tt[-1]+2)
    return tt