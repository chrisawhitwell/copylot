import numpy as np
def freqFP07(u):
    """
    Uses the relationship of Vachon \& Lueck 1984 detailed in Sommer's thesis on p.21 (and in their 2013 JTECH paper)
   
    Parameters
    ----------
    u : float
        mean flow speed past the sensor in m/s

    Returns
    -------
    fc : float
        Frequency Cutoff
    fo: float
        Unsure what this is!
    """
    to = 4.1e-3
    Wo = 1
    gam = -0.5
    
    tdp = to*(u/Wo)**gam
    fc = 1/(2*np.pi*tdp)
    fo = 25*u
    return fc,fo