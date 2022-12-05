import numpy as np
def bin_avg_spectra(f,Pxx,n=3,method='mean'):
    """
    Bin averages a PSD 
   
    Parameters
    ----------
    f : numpy array
        Single-sided power spectral density of velocity signal
    Pxx : numpy array
        Frequencies associated with Pw
    n : int 
        Number of points to include in each bin
    method :  string
        Method to use to calculate value fo each bin. Options include: 'mean' and 'median'
       
    Returns
    -------
    f : numpy array
        Returns TKE dissipation from IDM
    Pxx: numpy array
        Returns a dict with misfit criteria used in IDM process
    """
    
    if n%2 != 1:
        raise ValueError('Only bins with an odd number of points is currently supported')
    
    if f[0] == 0: ## Was mean removed from f/Pxx? If not, remove it.
        f = f[1:]
        Pxx = Pxx[1:]
    
    no_bins = int(len(Pxx)/n)
    
    c=1
    
    Pxx_bin = np.empty(no_bins)
    f_bin = np.empty(no_bins)
    
    
    for i in range(no_bins):
        if method == 'mean' : Pxx_bin[i] = np.mean(Pxx[c-1:c+2])
        elif method == 'median': Pxx_bin[i] = np.median(Pxx[c-1:c+2])
        else: raise ValueError('Method not recognised. Only "mean" and "median" currently supported.')
        f_bin[i] = f[c]
        c+= n
        
    return f_bin, Pxx_bin