import numpy as np

def despike(u,max_iter=20,window=None,intensity=1):
    """
    Goring & Nikora 2002 Phase Space Despike. 

    Ta @ Kilcher

    Parameters
    ----------
    u : np.array
        velocity sequence
    max_iter : int
        Number of iterations to loop through while despiking. Defaults to 20
    window :  int or None
        Window size to despike over. Defaults to the length of the whole record

    Returns
    ---------
    mask: np.array
        "True" elements of array is good data, "False" is bad data. Use like vel[mask]

    """

    n = len(u)
    x = np.arange(n)
    mask = np.ones_like(u).astype('bool')
    u_temp = np.copy(u)
    mask_temp = np.copy(mask)

    if not window:window=n
    L = int(np.ceil(n/window))

    count = 0
    
    while count < max_iter:
        ind = 0
        for i in range(L):
            mask_temp[ind:ind+window] = ~_phaseSpaceThresh(u_temp[ind:ind+window],intensity=intensity)
            ind += window
        if all(mask == ~mask_temp): break # no new spikes detected
        else:
            mask *= mask_temp
            u_temp = np.interp(x,x[mask],u_temp[mask])
            count += 1
    return mask

def _calcab(al, Lu_std_u, Lu_std_d2u):
    """
    Solve equations 10 and 11 of Goring+Nikora2002.

    Note: I took this directly from Kilcher. Ta Kilcher
    """
    return tuple(np.linalg.solve(
        np.array([[np.cos(al) ** 2, np.sin(al) ** 2],
                  [np.sin(al) ** 2, np.cos(al) ** 2]]),
        np.array([(Lu_std_u) ** 2, (Lu_std_d2u) ** 2])))

def _phaseSpaceThresh(u,intensity=1):
    """
    Implements the Goring+Nikora2002 despiking method, with Wahl2003
    correction.

    Note: I took this directly from Kilcher. Ta Kilcher
    """
    if u.ndim == 1:
        u = u[:, None]
    u = np.array(u)  # Don't want to deal with marray in this function.
    Lu = intensity * (2 * np.log(u.shape[0])) ** 0.5
    u = u - u.mean(0)
    du = np.zeros_like(u)
    d2u = np.zeros_like(u)
    # Take the centered difference.
    du[1:-1] = (u[2:] - u[:-2]) / 2
    # And again.
    d2u[2:-2] = (du[1:-1][2:] - du[1:-1][:-2]) / 2
    # d2u[2:-2]=np.diff(du[1:-1],n=2,axis=0) # Again, wrong.
    p = (u ** 2 + du ** 2 + d2u ** 2)
    std_u = np.std(u, axis=0)
    std_du = np.std(du, axis=0)
    std_d2u = np.std(d2u, axis=0)
    alpha = np.arctan2(np.sum(u * d2u, axis=0), np.sum(u ** 2, axis=0))
    a = np.empty_like(alpha)
    b = np.empty_like(alpha)
    for idx, al in enumerate(alpha):
        # print( al,std_u[idx],std_d2u[idx],Lu )
        a[idx], b[idx] = _calcab(al, Lu * std_u[idx], Lu * std_d2u[idx])
        # print( a[idx],b[idx] )
    if np.any(np.isnan(a)) or np.any(np.isnan(a[idx])):
        print('Coefficient calculation error')
    theta = np.arctan2(du, u)
    phi = np.arctan2((du ** 2 + u ** 2) ** 0.5, d2u)
    pe = (((np.sin(phi) * np.cos(theta) * np.cos(alpha) +
            np.cos(phi) * np.sin(alpha)) ** 2) / a +
          ((np.sin(phi) * np.cos(theta) * np.sin(alpha) -
            np.cos(phi) * np.cos(alpha)) ** 2) / b +
          ((np.sin(phi) * np.sin(theta)) ** 2) / (Lu * std_du) ** 2) ** -1
    pe[:, np.isnan(pe[0, :])] = 0
    return (p > pe).flatten('F')