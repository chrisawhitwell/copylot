import numpy as np
import copylot as cplt

def _batchelor_xfit(k,Pk,kd,visc,epsi,dof,tIDM=True):
    chir = np.array([10**(-12),10**(-1)])
    
    ## strip out leading 0 if required
    k = k[k>0]
    Pk = Pk[k>0]
    
    chis =  np.logspace(np.log10(chir[0]),np.log10(chir[1]),100)
    Pt =  modelTurbSpec(epsi,chis,kd,visc,k,tIDM=True)
    
    if np.max(Pk) > np.max(Pt[:]):
        return np.nan
    if np.max(Pk) <  np.min(Pt[:]):
        return np.nan
    
    logL,ind,a  =  cplt.turbulence.utils._logLikelihood(chis,Pk,Pt,dof)

    chis = np.logspace(np.floor(np.log10(chis[ind]))-1,np.ceil(np.log10(chis[ind]))+1,100)
    chis =  checkInRange(chis,chir)
    
    Pt =  modelTurbSpec(epsi,chis,kd,visc,k,tIDM=True)
    logL,ind,a  =  cplt.turbulence.utils._logLikelihood(chis,Pk,Pt,dof)
    
    ## Narrow search
    dec =  np.floor(np.log10(chis[ind]))
    emin =  (np.floor(chis[ind]/(10**dec))-0.1)*10**dec
    emax =  (np.ceil(chis[ind]/(10**dec))+0.1)*10**dec
    
    chis =  np.logspace(np.log10(emin),np.log10(emax),100)
    chis = checkInRange(chis,chir)
    
    Pt =  modelTurbSpec(epsi,chis,kd,visc,k,tIDM=True)
    logL,ind,a  =  cplt.turbulence.utils._logLikelihood(chis,Pk,Pt,dof)
    
    ### Final estimates and errors
    
    chiT =  chis[ind]
    
    if (ind + 2 > len(logL)-1) or (ind - 2 < 0):
        std_error =  10**10
    else:
        h = np.diff(chis)
        h = h[ind]
        df_logL = (-logL[ind+2] + 16*logL[ind+1] - 30*logL[ind] + 16*logL[ind-1] - logL[ind-2]/(12*h**2))
        st_error =  np.sqrt(-1/df_logL)
    
    return chiT

def checkInRange(chis,rrlim):
    if chis[0] < rrlim[0]:
        r = [rrlim[0]]
    else:
        r = [chis[0]]
    
    if chis[-1] > rrlim[1]:
        r.append(rrlim[1])
    else:
        r.append(chis[-1])
    r = np.asarray(r)
    
    chis = np.logspace(np.log10(r[0]),np.log10(r[1]),100)
    
    return chis

def modelTurbSpec(epsi,chis,kd,visc,k,tIDM=True):
    if tIDM:
        Ct =  0.4
        Pt =  Ct*k[:,None]**(1/3)@chis[None,:]*epsi**(-1/3)
    else:
        Pt = np.empty((kind.shape[0],chis.shape[0]))
        for i, chi in enumerate(chis):
            Pt[:,i] =  cplt.turbulence.utils._kraichnan_spec(epsi,chi,visc,kd,k=k)
    return Pt