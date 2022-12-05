import numpy as np

def _kraichnan_spec(epsi,chiT,nu=1.04e-06,kd=1.47e-07,q=5.25,k=None):
    """
    Returns theoretical Kraichnan spectra
    
    Parameters
    ----------
    epsi : float
        Dissipation of TKE
    chiT : float
        Dissipation of thermal variance
    nu : float
        Viscocity of seawater
    kd : float
        Thermal diffusivity of seawater
    q : float
        Degrees of freedom of Pw
    k : None (Default) or numpy.array
        Range of wavenumbers k to compute the Kraichnan spectra over
        
        
    Returns
    -------
    Pall : numpy.array
        Returns theoretical Kraichnan spectra

    
    Code is adapted from Bluteaus Matlab scripts.
    """

    knlim = 0.1
    Ct = 0.4

    kb = (epsi/((nu*kd**2)))**0.25
    eta = ((nu**3)/epsi)**0.25

    if  (type(k) != np.ndarray):
        if k == None:
            k = np.linspace(0.01*eta,kb,200)
        else:
            raise ValueError('k must either be a numpy array or None')
    
    y = np.sqrt(q)*k/kb
    
    Phi =  (y*np.sqrt(q)*chiT/(kb*kd))*np.exp(-np.sqrt(6)*y)
    Pic = _ic_dTdx_spec(k,epsi,chiT)
    
    ind = np.where(k*eta < knlim)[0]
    Pall = Phi
    Pall[ind] = Pic[ind]
    
    if (any(ind) & (k[-1]*eta>1.5*knlim)):
            ind  = (k*eta <=  2*knlim) & (k*eta >= knlim)
            if ind[0]: ind[0] = False # Set first element to false if not already
            
            if ind[-1]: ind[-1] = False # Set last element to false if not already
            
            Pall[ind] =  np.nan
            
            gdInd = np.array([np.where(ind)[0][0]-1,np.where(ind)[0][-1]+1])
            inter = np.interp(k[ind],k[gdInd],Pall[gdInd])
            Pall[ind] = inter
            
            ind = np.where(ind)[0]
            ind = np.arange(ind[-1]-5,ind[-1]+6)
            ind = check_krange(ind,len(Pall))
            
            for i in range(8):
                Pall[ind] =  smooth(Pall[ind])
    return Pall
    
def _ic_dTdx_spec(k,epsi,chiT,Ct=0.4):
    return Ct*chiT*(epsi**(-1/3))*(k**(1/3))

def smooth(a,WSZ=5):
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def check_krange(ind,nPts):
    tmpInd = np.where(ind>0)
    ind = ind[tmpInd]
    
    tmpInd =  np.where(ind < nPts)
    ind = ind[tmpInd]
    return ind