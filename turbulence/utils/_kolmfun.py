import numpy as np

def _kolmfun(epst,Pxx,fw,U,alp):
    Pt = alp*fw**(-5/3)*(epst*U)**(2/3)
    residuals = Pt-Pxx
    sse = np.sum(residuals**2)
    err = sse/(len(Pxx)-2)
    sst = np.sum((Pxx-np.mean(Pxx))**2)
    R2 = 1-(sse/sst)
    return err