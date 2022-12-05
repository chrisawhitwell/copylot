import xarray as xr
import numpy as np
from scipy import signal
from scipy import integrate

def _convert_FP07(da,calibration_dict=None):
    if calibration_dict == None:
        print('No calibration information given, using information for 002/937')
        calibration_dict = {'diff_gain':0.954,
                            'adc_zero':0,
                            'adc_fs':5.0,
                            'adc_bits':16,
                            'a':2.04794,
                            'b':1.0009,
                            'G':6,
                            'E_B':0.68237,
                            'beta_1':3032.74,
                            'beta_2':237708,
                            'T_0':286.353}
    
    diff_gain = calibration_dict['diff_gain']
    adc_zero = calibration_dict['adc_zero']
    adc_fs = calibration_dict['adc_fs']
    adc_bits = calibration_dict['adc_bits']
    a = calibration_dict['a']
    b = calibration_dict['b']
    G = calibration_dict['G']
    E_B = calibration_dict['E_B']
    beta_1 = calibration_dict['beta_1']
    beta_2 = calibration_dict['beta_2']
    T_0 = calibration_dict['T_0']

    #####
    dat = da.values

    f_c = 1/(2*np.pi*diff_gain)
    f_s = 10**9/(da.time.values[1] - da.time.values[0]).astype('timedelta64[ns]').astype('float64')
    filt = signal.butter(1,f_c,btype='low',output='ba',fs=f_s)

    timeV = np.arange(0,2*diff_gain,1/f_s)
    p = np.polyfit(timeV,dat[:len(timeV)],1)
    previousOutput = p[1] - diff_gain*p[0]
    #z = signal.lfiltic(b,a,np.array(previousOutput),np.array(dat[0]))
    T1_fast_dec_py = signal.lfilter(filt[0],filt[1],dat)

    Z = T1_fast_dec_py*(adc_fs/2**adc_bits) + adc_zero
    Z = ((Z-a)/b)*2/(G*E_B)
    Z[Z>0.6] = 0.6
    Z[Z<-0.6] = -0.6

    physical = (1-Z)/(1+Z)
    Log_R = np.log(physical)

    physical = 1/T_0 + (1/beta_1)*Log_R
    physical = physical + (1/beta_2)*Log_R**2
    physical = 1/physical - 273.15

    da_out = xr.DataArray(physical,\
                             dims=['time'],attrs={'long_name':'Temperature (Fast)','units':'Degrees C'})
    
    da_out.attrs.update(calibration_dict)
    return da_out

def _motion(ds,**kwargs):
    opts = {'acc_filter':0.1,'stitch':False,'inplace':True,'IMU_Data':False}
    opts.update(kwargs)
    if 'vel_filter' not in kwargs:
        opts['vel_filter'] = opts['acc_filter']/3

    L = np.array([0,0,-0.36]) - np.array([0.00635,0.00635,0.14986])

    ## Assign vars
    acc = ds['acc'].values
    omega = ds['omega'].values
    vel_raw = ds['vel'].values
    o_mat = ds['orient_mat'].values
    o_mat_T = np.transpose(o_mat,[1,0,2])

    ## Transform acc and vel into enu
    acc = np.einsum('ijk,j...k->i...k',o_mat_T,acc)
    vel_raw_enu = np.einsum('ijk,j...k->i...k',o_mat_T,vel_raw)
    ##### Calculate the translationally induced velocity
    ## Filter the accelerometer signal
    fs = ds.attrs['Sampling Rate [Hz]']
    fn = fs/2

    acc_filt = signal.bessel(1,opts['acc_filter'],btype='high',output='sos',fs=fs)
    vel_filt = signal.bessel(1,opts['vel_filter'],btype='high',output='sos',fs=fs)

    acc_hp =  signal.sosfiltfilt(acc_filt,acc)
    vel_t = integrate.cumtrapz(acc_hp,dx=1/fs,initial=0)
    if ('vel_filter',None) in opts.items():
        vel_t_hp = vel_t
    else:
        vel_t_hp = signal.sosfiltfilt(vel_filt,vel_t)

    ##### Calculate the rotationally induced velocity
    vel_r = np.array([
        (L[2] * omega[1,:] - L[1] * omega[2,:]),
        (L[0] * omega[2,:] - L[2] * omega[0,:]),
        (L[1] * omega[0,:] - L[0] * omega[1,:])])

    vel_r = np.einsum('ijk,j...k->i...k',o_mat_T,vel_r) # transform to ENU

    ##### Actually correct velocity signal
    vel = vel_raw_enu + vel_t_hp + vel_r

    ##### Stitch
    if ('stitch',True) in opts.items():
        if 'stitch_cut' in opts: cut = kwargs['stitch_cut']
        else: cut = acc_filter

        freqs = np.fft.rfftfreq(len(vel[0,:]),d=1/fs)
        freqs_low = (freqs < cut)

        for id in [0,1,2]:
            v_a = np.fft.rfft(vel[id,:])
            r_a = np.fft.rfft(vel_raw_enu[id,:])
            v_a[freqs_low] = r_a[freqs_low]
            vel[id,:] = np.fft.irfft(v_a)
    
    ##### Output
    if ('inplace',True) in opts.items():
        ds['vel'] = xr.DataArray(vel,dims=['coord','time'],attrs={'long_name':'Velocity','units':'m/s'})
        if ('IMU_Data',True) in opts.items():
            ds['vel_raw'] = xr.DataArray(vel_raw_enu,dims=['coord','time'],attrs={'long_name':'Raw Velocity','units':'m/s'})
            ds['vel_rot'] = xr.DataArray(vel_r,dims=['coord','time'],attrs={'long_name':'IMU Rotational Velocity','units':'m/s'})
            ds['vel_trans'] = xr.DataArray(vel_t_hp,dims=['coord','time'],attrs={'long_name':'IMU HP Translational Velocity','units':'m/s'})
            ds.attrs['IMU_Data'] = 'True'
        else:
            ds.attrs['IMU_Data'] = 'False'

        ds.attrs['Motion corrected'] = 'True'
        ds.attrs['IMU Accelerometer HP Filter'] = opts['acc_filter']
        ds.attrs['IMU Velocity HP Filter'] = opts['vel_filter']
        ds.attrs['Coordinate system'] = 'ENU'
    else:
        dss = ds.copy(deep=True)
        dss['vel'] = xr.DataArray(vel,dims=['coord','time'],attrs={'long_name':'Velocity','units':'m/s'})
        if ('IMU_Data',True) in opts.items():
            dss['vel_raw'] = xr.DataArray(vel_raw_enu,dims=['coord','time'],attrs={'long_name':'Raw Velocity','units':'m/s'})
            dss['vel_rot'] = xr.DataArray(vel_r,dims=['coord','time'],attrs={'long_name':'IMU Rotational Velocity','units':'m/s'})
            dss['vel_trans'] = xr.DataArray(vel_t_hp,dims=['coord','time'],attrs={'long_name':'IMU HP Translational Velocity','units':'m/s'})
            dss.attrs['IMU_Data'] = 'True'
        else:
            dss.attrs['IMU_Data'] = 'False'

        dss.attrs['Motion corrected'] = 'True'
        dss.attrs['IMU Accelerometer HP Filter'] = opts['acc_filter']
        dss.attrs['IMU Velocity HP Filter'] = opts['vel_filter']
        dss.attrs['Coordinate system'] = 'ENU'
        return dss