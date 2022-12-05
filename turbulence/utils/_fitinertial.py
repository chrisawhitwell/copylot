import numpy as np
from scipy import optimize
from ._kolmfun import _kolmfun

def _fitinertial(fw,Pxx,U,Fdir=1):
    alp = 1.5*(18/55)*Fdir
    b=np.mean(np.log10(Pxx)+(5/3)*np.log10(fw))
    start_point= (((10**b)/alp)**1.5)/U
    epsilon = optimize.fmin(_kolmfun,start_point,args=(Pxx,fw,U,alp),disp=0)
    
    Pt = alp*fw**(-5/3)*(epsilon*U)**(2/3)
    residuals = Pt-Pxx
    sse = np.sum(residuals**2)
    err = sse/(len(Pxx)-2)
    sst = np.sum((Pxx-np.mean(Pxx))**2)
    R2 = 1-(sse/sst)
    
    return epsilon,R2,err
