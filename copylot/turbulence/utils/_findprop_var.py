from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import numpy as np

def _findprop_var(km,epsi,nu,kd=1e-7):
    """
    Returns proporation of variance resolved of temperature spectra
    
    Parameters
    ----------
    km  :  float
        Maximum wavenumber resolved for integration 
    epsi : float
        Dissipation of TKE
    nu : float
        Viscocity of seawater
    kd : float
        Thermal diffusivity of seawater
    
    Returns
    -------
    prop : float
        Proportion of variance

    
    Code is adapted from Bluteaus Matlab scripts.
    """
    chiF =  1e-6
    kb =  (epsi/(nu*kd**2))**0.25
    k = np.logspace(0,np.log10(2*kb))
    
    Pall = cplt.turbulence.utils._kraichnan_spec(1e-8,chiF,nu,kd,k=k)
    
    frac =  6*kd*cumulative_trapezoid(Pall,k,initial=0)/chiF
    x = k/kb
    
    ind =  np.where(frac >=1 )[0]
    frac[ind] = 1
    
    xm = km/kb
    
    interp_f =  interp1d(x,frac,fill_value=np.nan,bounds_error=False)
    prop = interp_f(xm)
    
    
    if np.isnan(prop):
            interp_f =  interp1d(x,frac,kind='nearest',fill_value="extrapolate")
            prop = interp_f(xm)
    return prop
    