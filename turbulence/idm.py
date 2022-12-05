import numpy as np
import matplotlib.pyplot as plt
from .utils._epsiQAQCcalcs import _epsiQAQCcalcs
from .utils._specPortion_check import _specPortion_check
from .utils._assign_misfit import _assign_misfit

def idm(v1_mean,Pw,fw,vel_dir,dof,dec=0.7,low_freq_lim=None,high_freq_lim=None,visc=1e-6,min_R2=0,NS=None,knlim=0.1,plot=False,verbose=False):
    """
    Estimates TKE dissipation from the IDM method outlined in Bluteau et al. 2011
    
    Parameters
    ----------
    v1_mean : int
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    Pw : array
        Single-sided power spectral density of velocity signal with units (m/s)^2/(rad/s)
    fw : array
        Frequencies (in rad/s) associated with Pw
    veldir : int
        Flow direction (1:Streamwise,2/3:Cross-stream)
    dof : int
        Degrees of freedom of Pw
    dec : int, optional
        Minimum number of decades used for fitting
        Default: 0.7
    low_freq_lim : int, optional
        Excludes frequencies below low_freq_lim 
        Default: None
    high_freq_lim : int, optional
        Excludes frequencies above high_freq_lim
        Default: None
    visc : int, optional
        Kinematic viscocity of water
        Default: 1e-6
    min_R2 : int, optional
        min_R2 of fit to accept spectra segment
        Default: 0
    NS : int, optional
        Shear or stratification for low wave number limit (in rad/s)
        Default: None
    knlim : int, optional
        Default: 0.1
    plot : bool, optional
        If True, returns plot and axes
        
    Returns
    -------
    epsilon : numpy.float64
        Returns TKE dissipation from IDM
    misfit: dict
        Returns a dict with misfit criteria used in IDM process
    axes: array of matplotlib.axes
        Returns axes object if plot=True
    
    Code is adapted from Bluteaus Matlab IDM scripts.
    """

    brange = np.array([-0.75,0.75]) -5/3 
    kcte = 1.5 # Kolmogorov Constant
    fact = 10**dec

    if vel_dir == 1:
        F_dir = 1
    elif vel_dir == 2:
        F_dir = 4/3
    elif vel_dir == 3:
        F_dir = 4/3
    else:
        raise ValueError('Vel_dir should be an int in 1,2 or 3')

    s_sq_best = 1e10
    epsilon = np.nan
    best_ind = np.array([np.nan,np.nan])
    misfit = dict()
    Specfit = dict()


    if any(np.isnan(fw)):
        raise ValueError

    fwall = fw[fw>0]
    Pwall = Pw[fw>0]

    if not low_freq_lim:
        low_freq_lim = np.min(fwall)

    if not high_freq_lim:
        high_freq_lim = np.max(fwall)

    ind =  (fw >= low_freq_lim) & (fw <= high_freq_lim)

    Specfit['P'] = Pw[ind]
    Specfit['f'] = fw[ind]
    Nb = len(Specfit['f'])

    ## Search for range of data with best fit
    # Forward search entire dataset

    cc = 0

    mnt = np.where((Specfit['f'] > fact*Specfit['f'][0]))[0]

    if mnt.size > 0:mn = mnt[0]
    else:mn = Nb

    tt = 0
    loop_count = 0 
    while mn.size > 0:
        #print(loop_count)
        loop_count+=1
        tt = tt+1
        ind = np.arange(cc,mn)
        cc += 1

        mn = np.where((Specfit['f'] > fact*Specfit['f'][cc]))[0]
        if mn.size > 0:
            mn = mn[0]

        epsilon,misfit = _epsiQAQCcalcs(Specfit['f'][ind],Specfit['P'][ind],v1_mean,F_dir,dof,visc,NS)
        misfit['dec'] = dec

        if tt==1:allmisfit = misfit
        else:allmisfit = _assign_misfit(misfit,allmisfit)

        tcont = _specPortion_check(misfit['R2'],min_R2,[misfit['kn'],knlim],misfit['kL'],np.append(brange,misfit['slope']))

        if tcont != 1:
            if misfit['MAD'] < s_sq_best:
                s_sq_best = misfit['MAD']
                best_ind = ind

    if any(np.isnan(best_ind)):
        if verbose:
            print('No inertial subrange satisfying all requirements in _specPortion_check was satisfied.')
        epsilon = np.nan
        misfit = dict()
        return epsilon,misfit, best_ind

    ## Now search backwards from half of dec (min decade requirement) of last point in best_ind

    fhd = 10**(dec/2)
    Nbmax = np.where((Specfit['f'] > fhd*Specfit['f'][best_ind[-1]]))[0]

    if Nbmax.size > 0:Nbmax = Nbmax[0]
    else:Nbmax = Nb

    mn1 = np.where(Specfit['f']*fhd < Specfit['f'][best_ind[-1]])[0]
    if mn1.size > 0: mn1 = mn1[-1]
    else: mn1 = 0
    mn = np.where(Specfit['f']*fact < Specfit['f'][Nbmax-1])[0]
    if mn.size > 0: mn = mn[-1]
    else: mn = 0


    cc = 0
    s_sq_back = 1e10
    back_ind=best_ind
    tt = len(allmisfit['fw'])

    ## Backsearching restricted area
    while mn.size > 0:
        loop_count+=1
        tt = tt+1
        ind = np.arange(mn,Nbmax-cc)

        cc=cc+1

        mn = np.where(Specfit['f']*fact < Specfit['f'][Nbmax-cc])[0]
        if mn.size > 0:
            mn = mn[-1]

        epsilon, misfit = _epsiQAQCcalcs(Specfit['f'][ind],Specfit['P'][ind],v1_mean,F_dir,dof,visc,NS)
        tcont = _specPortion_check(misfit['R2'],min_R2,[misfit['kn'],knlim],misfit['kL'],np.append(brange,misfit['slope']))

        if tcont != 1:
            if misfit['MAD'] < s_sq_best:
                s_sq_back = misfit['MAD']
                back_ind = ind

        if mn<mn1:break

    # Incremental search around best range found above

    i1 = max([best_ind[0],back_ind[0]]) + 1
    i2 = min([best_ind[-1],back_ind[-1]]) - 1

    if i2<i1: i2=i1+1

    if s_sq_back < s_sq_best: best_ind = back_ind


    ######
    if verbose:
        print('Searched ' + str(tt) + ' portions of spectra')

    if all(~np.isnan(best_ind)):
        epsilon,misfit = _epsiQAQCcalcs(Specfit['f'][best_ind],Specfit['P'][best_ind],v1_mean,F_dir,dof,visc,NS)

    axes = None
    #### Plotting Section ####
    if plot == True:
        fig,axes = plt.subplots(figsize=(16,12),nrows=6,sharex=True,gridspec_kw={'height_ratios':[3,1,1,1,1,1]})
        alpha = F_dir*1.5*18/55

        ## Top plot ##
        axes[0].loglog(fwall,Pwall,ls='-',marker='.',label='Raw Spectra')

        if ~np.isnan(epsilon):
            ft = fw[best_ind]
            Pt = alpha*(epsilon*v1_mean)**(2/3)*ft**(-5/3)

            pp = np.where(allmisfit['fw'] >=np.median(ft))[0][0]
            medft = allmisfit['fw'][pp]

            Ps = allmisfit['Apower'][pp]*ft**(allmisfit['slope'][pp])

            axes[0].loglog(ft,Pt,label='Theoretical Spectra',lw = 1.5,ls='--')
            axes[0].loglog(ft,Ps,label='Fitted Spectra',lw = 1.5, ls='--')

            eta = (visc**(3)/epsilon)**0.25
            kn = eta*axes[0].get_xticks()/v1_mean
            Ekn = axes[0].get_yticks()*v1_mean/(epsilon*visc**5)**0.25
            ### Need to finish stuff here!

        axes[0].set_ylabel(r'$\Phi [\frac{(m/s)^2}{rad/s}]$')
        axes[-1].set_xlabel(r'$f [rad/s]$')

        ## Second plot ##

        axes[1].semilogy(allmisfit['fw'],allmisfit['epsilon'],marker='.')
        axes[1].plot(medft,allmisfit['epsilon'][pp],'o')
        axes[1].set_ylabel('$\epsilon$')

        ## Third plot ##

        axes[2].plot(allmisfit['fw'],allmisfit['slope'])
        axes[2].plot(medft,allmisfit['slope'][pp],marker='o')
        axes[2].set_ylabel('Slope')

        ## Fourth plot ##

        axes[3].semilogy(allmisfit['fw'],allmisfit['MAD'])
        axes[3].semilogy(allmisfit['fw'],allmisfit['var'])
        axes[3].plot(medft,allmisfit['MAD'][pp],marker='o')
        axes[3].plot(medft,allmisfit['var'][pp],marker='o')
        axes[3].set_ylabel('MAD & Var')

        ## Fifth plot ##

        axes[4].semilogy(allmisfit['fw'],allmisfit['kL'])
        axes[4].plot(medft,allmisfit['kL'][pp],marker='o')
        axes[4].set_ylabel('kL')

        ## Bottom plot ##

        axes[5].plot(allmisfit['fw'],allmisfit['R2'])
        axes[5].plot(medft,allmisfit['R2'][pp],marker='o')
        axes[5].set_ylabel('$R^2$')
        return epsilon,misfit,best_ind,axes
    else:
        return epsilon,misfit,best_ind
 