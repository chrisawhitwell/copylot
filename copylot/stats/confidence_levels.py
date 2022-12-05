from scipy import stats

def confidence_levels(p,N,M,window='hann'):
    """
    Parameters
    ----------
    p: float
        Confidence interval expressed as fraction
    N: int
        Number of data points in time-series used for spectra
    M: float
        Half width of window in time-domain

    Returns
    -------
    v : float
        Effective degrees of freedom
    CL: tuple
        Condifence level for max, min and median. Multiply these three values by the y value where you want the error to appear.
    
    Code is adapted from Bluteaus Matlab scripts.
    """
    if window == 'hann':
        v = 8*N/(3*M)
    else:
        print('Pls code in this window, see E&T section 5.4.8 "Confidence Intervals on Spectra"')
        raise ValueError
    
    alpha = (1-p)/2
    CLmax = (v/stats.chi2.ppf(alpha,v))
    CLmin = (v/stats.chi2.ppf(1-alpha,v))
    CLmed = (v/stats.chi2.ppf(0.5,v))
    
    return v,(CLmax,CLmin,CLmed)