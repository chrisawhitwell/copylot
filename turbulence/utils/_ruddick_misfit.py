import numpy as np

def _ruddick_misfit(Pw,Pt):
    MAD2 = (1/len(Pt))*np.sum(abs((Pw/Pt)-np.mean(Pw/Pt)))
    varm = np.var(Pw/Pt)
    return varm,MAD2