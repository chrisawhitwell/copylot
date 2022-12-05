import numpy as np
from scipy import signal

def psd(x,fs,f_units='Hz'):
    """
    Calculates the one-sided power spectral density (psd) of a signal.
    Parameters
    -------------
    x: signal
    fs: sample frequency

    Returns
    -------------
    freq: numpy array
        frequency terms for psd 
    psd: numpy array
        Power Spectral Density 
    """
    x = signal.detrend(x)
    n = len(x)
    T = n/fs
    a = np.fft.rfft(x)
    psd = 2*((1/fs)**2)*(1/T)*np.abs(a)**2
    freq = np.fft.rfftfreq(n,1/fs)

    if f_units == 'Hz':
        psd_out = psd
        freq_out = freq
    elif f_units == 'rad/s':
        psd_out = psd/(2*np.pi)
        freq_out = freq*2*np.pi
    else:
        raise ValueError('f_units not recognised')

    return freq_out,psd_out