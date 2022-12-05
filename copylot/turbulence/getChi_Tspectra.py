import numpy as np
import copylot as cplt

def chi(epsi,k,Pk,visc,kd,kNy=None,dof=None,fastR=1.1):
    """
    Returns chi from the temperature gradient spectra by fitting to the inertial subrange and integrating
    
    Parameters
    ----------
    epsi : float
        Dissipation of TKE
    k : np.array
        Wavenumbers in rad/m
    Pk : np.array
        dTdx spectra in (degC/m)^2 / (rad/m)
    visc: float
        Viscocity of seawater
    kd : float
        Thermal diffusivity of seawater
    kNy : float or None (default)
        Nyquist wavenumber for fit. If "None", defaults to min(k)
    dof : int
        Degrees of freedom for spectra
    fastR: int
        Factor for speeding up fitting by stepping through multiple indices in for each fit
        
        
    Returns
    -------
    chiF : numpy.array
        Returns theoretical Kraichnan spectra
    theoSpec: dict
        Dictionary containing the theoretical spectra used for the fit
    misft: dict
        Dictionary containing the misfit criteria (MAD and Var)

    
    Code is adapted from Bluteaus Matlab scripts.
    """
    dec = 0.5
    knlim = 0.1
    
    tIDM = True
    
    if not kNy:
        kNy =  np.max(k)
    
    chiF = np.nan
    MAD2 = np.nan
    misft = {}
    misft['MAD2'] = MAD2
    
    theoSpec = {}
    theoSpec['k'] = k
    theoSpec['Pk'] = cplt.turbulence.utils._kraichnan_spec(epsi,1e-6,visc,kd,k=k)
    
    eta =  ((visc**3)/epsi)**0.25
    kb =  (epsi/(visc*kd**2))**0.25
    
    ### Here is where to calculate integrated Chi!
    
    fact = 10**dec
    misft['madRej'] =  2*np.sqrt(2/dof)
    
    chiSweep = {}
    chiSweep['chiT'],chiSweep['kmed'],misfit =  cplt.turbulence.utils._sweepIDM_Chi(Pk,k,epsi,dof,visc,kd,dec,kNy,fastR)
    
    if all(np.isnan(chiSweep['chiT'])):
        print('No fit within prescribed MLE range')
        return np.nan,np.nan,np.nan,np.nan
    
    tt =  np.argwhere((misfit['MAD2'] < misft['madRej'])&(chiSweep['kmed']*eta<knlim))
    if tt.size == 0:
        return np.nan,np.nan,np.nan,np.nan
    
    tt = tt[0]
    
    kmed =  chiSweep['kmed'][tt]
    ind = cplt.turbulence.utils._get_kInd(k,kmed/(0.5*fact),knlim/eta)
    
    chiF  =  cplt.turbulence.utils._batchelor_xfit(k[ind],Pk[ind],kd,visc,epsi,dof)
    theoSpec['k'] = k[ind]
    theoSpec['Pk'] =  cplt.turbulence.utils._kraichnan_spec(epsi,chiF,visc,kd,k=k[ind])
    
    temp, misft['MAD2'] = cplt.turbulence.utils._ruddick_misfit(Pk[ind],theoSpec['Pk'])
    
    return chiF,theoSpec,misft,chiSweep