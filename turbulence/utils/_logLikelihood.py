import numpy as np
from scipy import stats

def _logLikelihood(epst,Pk,Pt,v):
    nPxx = np.tile(Pk[:,None],(1,len(epst)))
    z = v*nPxx/Pt
    Z = stats.chi2.pdf(z,v)
    Y = np.log(Z/Pt)
    logL = np.sum(Y,0) + len(Pk)*np.log(v)
    val = np.nanmax(np.exp(logL-np.nanmax(logL)))
    ind = np.argmax(np.exp(logL-np.nanmax(logL)))
    return logL,ind,val