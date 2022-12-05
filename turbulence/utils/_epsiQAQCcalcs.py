import numpy as np
from ._fitinertial import _fitinertial
from ._MLE_IDM import _MLE_IDM
from ._fitpowerlaw import _fitpowerlaw
from ._ruddick_misfit import _ruddick_misfit
from ._calc_L import _calc_L

def _epsiQAQCcalcs(fw,Pw,v1_mean,F_dir,dof,visc,NS=None):
    mfields =  ['epsn1','R2','err','var','MAD','Apower','slope','slopeR2','slope_min','slope_max','kn','fw','npts','kL','epsilon']
    misfit = dict()
    
    misfit['epsn1'],misfit['R2'],err = _fitinertial(fw,Pw,v1_mean,F_dir)
    epsilon, misfit['err'] = _MLE_IDM(fw,Pw,v1_mean,F_dir,dof)
    
    alp = 1.5*(18/55)*F_dir
    Pt = alp*(epsilon*v1_mean)**(2/3)*fw**(-5/3)
    
    misfit['var'],misfit['MAD'] = _ruddick_misfit(Pw,Pt)
    
    misfit['Apower'],misfit['slope'],Aci,slope_ci,misfit['slopeR2'] = _fitpowerlaw(fw,Pw)
    
    misfit['slope_min'] = slope_ci[0]
    misfit['slope_max'] = slope_ci[1]
    
    misfit['kn'] = np.median((fw/v1_mean)*((visc**3)/epsilon)**0.25)
    
    if NS != None:
        L = _calc_L(epsilon,NS)
        k = fw/v1_mean
        kL = k*L
    else:
        kL = np.empty_like(fw)
        kL[:] = np.nan
    
    misfit['fw'] = np.median(fw)
    misfit['npts'] = len(fw)
    misfit['kL'] = np.median(kL)
    misfit['epsilon'] = epsilon
    
    for field in mfields:
        if all(t != field for t in list(misfit.keys())):
            misfit[field] = np.nan
    
    return epsilon, misfit