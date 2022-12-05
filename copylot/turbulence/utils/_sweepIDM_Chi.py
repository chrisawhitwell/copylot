import copylot as cplt
import numpy as np
def _sweepIDM_Chi(Pk,k,epsi,dof,visc,kd,dec,kNy,fastR):
    tIDM = True
    fact = 10**dec
    tfast =  True

    ## Strip out leading zero if exists
    if not kNy: kNy=np.max(k)

    k = k[k>0]
    Pk = Pk[k>0]

    Nb = len(k)

    mn = np.argwhere(k > fact*k[0])
    if mn.size ==  0: mn = np.asarray(Nb)
    else: mn = mn[0]

    misfit = {'var':[],'MAD2':[]}    
    kmed = []
    chiT = []
    
    cc = 0
    tt = -1

    while mn.size != 0 :
        mn = mn[0]
        tt += 1
        ind =  np.arange(cc,mn+1)
        ind = cplt.turbulence.utils._get_kInd(k,k[cc],k[mn])
        temp_kmed = np.median(k[ind])
        kmed.append(temp_kmed)
    
        tempChi = cplt.turbulence.utils._batchelor_xfit(k[ind],Pk[ind],kd,visc,epsi,dof,tIDM)
        chiT.append(tempChi)
        if np.isnan(tempChi):
            misfit['var'].append(np.nan)
            misfit['MAD2'].append(np.nan)
        else:
            Pt =  cplt.turbulence.utils._kraichnan_spec(epsi,tempChi,visc,kd,k=k[ind])
            tempVar,tempMAD2 =  cplt.turbulence.utils._ruddick_misfit(Pk[ind],Pt)
            misfit['var'].append(tempVar)
            misfit['MAD2'].append(tempMAD2)
    
        if temp_kmed > kNy: break
    
        cc += 1
    
        if tfast:
            if k[cc] <= fastR*k[cc-1]:
                cc = np.argwhere(k> fastR*k[cc-1])[0][0]
        mn = np.argwhere(k > fact*k[cc])
    kmed = np.asarray(kmed)
    chiT = np.asarray(chiT)

    misfit['var'] = np.asarray(misfit['var'])
    misfit['MAD2'] = np.asarray(misfit['MAD2'])

    return chiT,kmed,misfit