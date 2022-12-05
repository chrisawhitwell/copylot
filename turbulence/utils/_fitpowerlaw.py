import numpy as np
from scipy import optimize,stats

def _fitpowerlaw(xi,yi,bf=None,alp=0.1):
    # fit power law: y=Ax^b
    p = np.polyfit(np.log10(xi),np.log10(yi),1)
    p[1] = 10**p[1]
    
    if bf == None:
        _fun = lambda x,b1,b2: b2*x**b1
    else:
        b = bf
        _fun = lambda x,b2:b2*x**b
        p = p[1]
    try:
        popt,pcov = optimize.curve_fit(_fun,xi,yi,p)
    except RuntimeError:
        print('Curve fitting failed')
        return np.nan,np.nan,[np.nan,np.nan],[np.nan,np.nan],np.nan
    
    residuals = yi- _fun(xi, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yi-np.mean(yi))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    v = len(residuals) - len(popt)
    
    se = np.sqrt(np.diag(pcov))
    delta = se*stats.t.ppf(1-alp/2,v)
    
    if bf == None:
        A = popt[1]
        b = popt[0]
        Aci = [A-delta[1],A+delta[1]]
        bci = [b -delta[0],b+delta[0]]
    else: # Not sure if this works! Good luck
        A = popt[0]
        Aci = [A-delta[1],A+delta[1]]
        bci = None
        r2 = None
    return A,b,Aci,bci,r2