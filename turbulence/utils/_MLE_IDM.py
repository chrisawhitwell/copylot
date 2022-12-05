import numpy as np
from ._logLikelihood import _logLikelihood
from scipy import stats

def _MLE_IDM(fw,Pxx,U,Fdir,v):
    misfit = dict()
    alp=1.5*(18/55)*Fdir
    visc=1e-6
    fw = fw[fw>0]
    Pxx = Pxx[fw>0]
    U=abs(U)
    
    logfw = np.log10(fw)
    logPxx = np.log10(Pxx)
    
    btmp = logPxx + (5/3)*logfw
    b = stats.norm.ppf([0.05,0.95],loc=np.median(btmp),scale=np.std(btmp))
    
    epslim = (((10**b)/alp)**1.5)/U
    if epslim[0]<0:
        epslim[0] = 1e-15

    ### Initial Search
    epst = np.linspace(epslim[0],epslim[1],100)
    Pt = alp*(fw**(-5/3))[:,None]@((epst*U)**(2/3))[None,:]
    logL, ind, val = _logLikelihood(epst,Pxx,Pt,v)
    
    ### Narrow search to nearest 2 decade range
    epst=np.logspace(np.floor(np.log10(epst[ind]))-1, np.ceil(np.log10(epst[ind]))+1,100)
    Pt = alp*(fw**(-5/3))[:,None]@((epst*U)**(2/3))[None,:]
    logL, ind, val = _logLikelihood(epst,Pxx,Pt,v)
    
    ### Narrow search to nearest 0.2 decade range
    dec = np.floor(np.log10(epst[ind]))
    emin = (np.floor(epst[ind]/10**dec)-0.1)*10**dec
    emax = (np.ceil(epst[ind]/10**dec)-0.1)*10**dec
    
    epst = np.logspace(np.log10(emin), np.log10(emax),100)
    Pt = alp*(fw**(-5/3))[:,None]@((epst*U)**(2/3))[None,:]
    logL, ind, val = _logLikelihood(epst,Pxx,Pt,v)
    epsilon = epst[ind]
    
    if ((ind+2)>len(logL)-1) or ((ind-2)<0):
        std_error = 10**10
    else:
        h = np.diff(epst)
        h = h[ind]
        df_logL = (-logL[ind+2]+16*logL[ind+1]-30*logL[ind]+16*logL[ind-1]-logL[ind-2])/(12*h**2)
        std_error = np.sqrt((-1/df_logL))
    
    return epsilon,std_error