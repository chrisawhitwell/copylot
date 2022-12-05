from scipy import signal
import numpy as np

def correct_scalar_spectra(f,fs,diff_gain=1):
    """
    Uses the relationship of Vachon \& Lueck 1984 detailed in Sommer's thesis on p.21 (and in their 2013 JTECH paper)
   
    Parameters
    ----------
    f : numpy array
        Frequency of spectra in Hz
    fs : float
        Sampling frequency
    diff_gain : float
        Diff gain from uSquid config file

    Returns
    -------
    diffCorr : numpy array
        Correction for using 1st order numerical time-differentiated (eg diff.m) to get the scalar gradients
    convCorr: numpy array
        Correction for the recording continuous domain and a de-convolution in the discrete domain	(needed for RSI's  instruments)
    """
    diffCorr = np.ones_like(f)
    convCorr = np.ones_like(f)
    
    diffCorr[1:] = ((np.pi*f[1:])/(fs*np.sin(np.pi*f[1:]/fs)))**2
    
    b,a = signal.butter(1,1/(2*np.pi*diff_gain*fs/2))
    junk = abs(signal.freqz(b,a,f[:-1],fs=fs)[1]**2)
    H = 1+ (2*np.pi*f[:-1]*diff_gain)**2
    H = 1/H
    convCorr[:-1] = H/junk
    convCorr[-1] = convCorr[-2]
    return diffCorr,convCorr   