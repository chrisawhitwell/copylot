import xarray as xr
import numpy as np
from scipy import signal

from .accessor_funcs import _convert_FP07,_motion
from ..signals import despike as _despike

@xr.register_dataarray_accessor("xrADV")
class xrADV_da:
    def __init__(self,da):
        self.da = da
    def convert_FP07(self,**kwargs):
        '''
        Convert signal from FP07 into physical units

        Parameters
        ----------
        calibration_dict : dict or None
            Dictionary of calibration params for conversion. Defaults to my RS2019 instrument.
            Keys include: 'diff_gain','adc_zero','adc_fs','adc_bits','a','b','G','E_B','beta_1','beta_2','T_0'}

        DataArray requirements
        ----------
        Input needs to be 1D with a 'time' coordinate in np.datetime64

        Returns
        -------
        Dast temperature data : xr.DataArray


        Code is adapted from Bluteaus Matlab scripts and Rockland ODAS scripts.
        '''
        return _convert_FP07(self.da,**kwargs)

    def despike(self,**kwargs):
        """
        Generates a 1D boolean mask with coordinate time. Input should be (3 x time) velocity xr.DataArray
        ----------
        max_iter : int
            Max number of iterations
        window :  int
            Number of points used to calculate sigma
        intensity
            Scale universal threshold
  
        Returns
        -------
        qaqc mask: xarray dataarray
            DataArray mask
        """
        opts = {'max_iter':5, 'window':None, 'intensity':1} # Defaults
        opts.update(kwargs)
        opts['type'] = 'despike'
        m1 = _despike(self.da.values[0,:],**kwargs)
        m2 = _despike(self.da.values[1,:],**kwargs)
        m3 = _despike(self.da.values[2,:],**kwargs)
        mask =  m1 & m2 & m3 # these are for each beam but we dont have redundency so if one fails, drop all
        return xr.DataArray(mask,coords={'time':self.da.time.values},dims=['time'],attrs=opts)

    def correlation_qaqc(self,**kwargs):
        """
        Generates a 1D boolean mask with coordinate time. Input should be (3 x time) velocity xr.DataArray
        ----------
        threshold : int
  
        Returns
        -------
        qaqc mask: xarray dataarray
            DataArray mask
        """
        opts = {'thresh':70}
        opts['type'] = 'correlation'

        opts.update(kwargs)
        mask =  (self.da[0,:] > opts['thresh'])&(self.da[1,:] > opts['thresh'])&(self.da[2,:] > opts['thresh'])
        return mask

@xr.register_dataset_accessor("xrADV")
class xrADV_ds:
    def __init__(self,ds):
        self.ds = ds
    def motion(self,**kwargs):
        """
        Performs motion correction of ADV dataset using Kilcher (2017)
        ----------
        inplace : bool
            If True (default), return in place. Otherwise return copy ds
        acc_filter: float
            IMU accelerations high pass filter
        stitch: bool
            Remove low frequency IMU drift by replacing with raw velocimeter data 
        vel_filter : float
            IMU velocity high pass filter
        stitch_cut: float
            Frequency cutoff for replacing low frequency content of mc'd spectra
        IMU_Data : bool
            If True, keep raw IMU data 

        Returns
        -------
        ds: xarray dataset
            Dataset with velocities motion corrected


        """
        return _motion(self.ds,**kwargs)

    def transform(self,to,inplace=True):
        """
        This function is used to transform the velocities in an xrADV-described dataset 
        between coordinate systems.
        It requires that the dataset has:
        Global attribute:
            - 'Coordinate system' : 'XYZ', 'BEAM' or 'ENU'
        Data variable:
            - trans_mat : 3x3 matrix output from Vector
            - orient_mat : 3x3xtime matrix output from Vector with IMU

        Input: 
            - An xarray dataset in xrADV format
            - to :  a string 'XYZ','BEAM' or 'ENU' representing the coordinate system you
              wish to transform to.

        Output:
            - An xarray dataset with velocities transformed to new coordinate system, and labels 
              updated

        Labels:
            'XYZ':  x -> 'u' , y -> 'v' , z -> 'w'
            'BEAM': 1 -> '1' , 2 -> '2' , 3 -> '3'
            'ENU':  East -> 'E' , North -> 'N' , Up -> 'U'

        """
        if inplace: dss = self.ds
        else: dss = self.ds.copy(deep=True)

        if dss.attrs['Coordinate system'] == to:
            print('Dataset is already in this coordinate system')
        else:
            if to == 'BEAM':
                if dss.attrs['Coordinate system'] == 'XYZ': dss = XYZ2BEAM(dss)
                elif ds.attrs['Coordinate system'] == 'ENU': dss = XYZ2BEAM(ENU2XYZ(dss))
                elif ds.attrs['Coordinate system'] == 'PRINC': dss = XYZ2BEAM(ENU2XYZ(PRINC2ENU(dss)))
                else: raise ValueError

            elif to == 'XYZ':
                if dss.attrs['Coordinate system'] == 'BEAM': dss = BEAM2XYZ(dss)
                elif dss.attrs['Coordinate system'] == 'ENU': dss = ENU2XYZ(ds)
                elif dss.attrs['Coordinate system'] == 'PRINC': dss = PRINC2ENU(ENU2XYZ(ds))
                else: raise ValueError

            elif to == 'ENU':
                if dss.attrs['Coordinate system'] == 'BEAM': dss = XYZ2ENU(BEAM2XYZ(dss))
                elif dss.attrs['Coordinate system'] == 'XYZ': dss = XYZ2ENU(dss)
                elif dss.attrs['Coordinate system'] == 'PRINC': dss = PRINC2ENU(dss)
                else: raise ValueError

            elif to == 'PRINC':
                if dss.attrs['Coordinate system'] == 'BEAM': dss = ENU2PRINC(XYZ2ENU(BEAM2XYZ(dss)))
                elif dss.attrs['Coordinate system'] == 'XYZ': dss = ENU2PRINC(XYZ2ENU(dss))
                elif dss.attrs['Coordinate system'] == 'ENU': dss = ENU2PRINC(dss)
                else: raise ValueError
            else:
                print('The coordinate system '+ str(to) + ' is not recognised')
        if not inplace:return dss

    def _tilt_vel(self,inplace=True):
        """
        Correct my faulty tilt sensor. You dont want to use this.
        """
        if inplace: dss = self.ds
        else: dss = self.ds.copy(deep=True)

        time = dss['time'].values
        rad = -np.radians(3.36)
        rotmat = np.array([[np.cos(rad),0,-np.sin(rad)],[0,1,0],[np.sin(rad),0,np.cos(rad)]])
        vel = rotmat @ dss['vel'].values
        dss['vel'] = xr.DataArray(vel,coords={'coord':['x','y','z'],'time':time},dims=['coord','time'],attrs={'long_name':'Velocity (XYZ)','units':'m/s'})
        if not inplace: return dss

### Transformations
def BEAM2XYZ(ds):
    time = ds['time'].values
    ds['vel'] = xr.DataArray(ds['trans_mat'].values @ ds['vel'].values,coords={'coord':['x','y','z'],'time':time},dims=['coord','time'],attrs={'long_name':'Velocity (instrument)','units':'m/s'})
    ds.attrs['Coordinate system'] = 'XYZ'
    return dss

def XYZ2ENU(ds):
    time = ds['time'].values
    OMAT = ds['orient_mat'].values
    OMAT_T = np.transpose(OMAT,[1,0,2])
    vel_enu =  np.einsum('ijk,j...k->i...k',OMAT_T,ds['vel'].values)
    ds['vel'] = xr.DataArray(vel_enu,coords={'coord':['x','y','z'],'time':time},dims=['coord','time'],attrs={'long_name':'Velocity (ENU)','units':'m/s'})
    ds.attrs['Coordinate system'] = 'ENU'
    return ds

def ENU2PRINC(ds):
    vec = ds['vel']
    mean = vec.mean('time')
    theta = - np.arctan2(mean.sel(coord='y').values,mean.sel(coord='x').values)
    rot_mat = np.eye(3)
    rot_mat[:2,:2] = np.asarray([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    time = ds['time'].values
    vecs_t = rot_mat@ds['vel'].values
    ds['vel'] = xr.DataArray(vecs_t,coords={'coord':['x','y','z'],'time':time},dims=['coord','time'],attrs={'long_name':'Velocity (PRINC)','units':'m/s','Principle Angle (rad CW of East)':theta})

    if 'IMU_Data' in ds.attrs:
        if ds.attrs['IMU_Data'] == 'True':
            vel_raw = rot_mat@ds['vel_raw'].values
            ds['vel_raw'] = xr.DataArray(vel_raw,coords={'coord':['x','y','z'],'time':time},dims=['coord','time'],attrs={'long_name':'Raw Velocity (PRINC)','units':'m/s','Principle Angle (rad CW of East)':theta})
            vel_rot = rot_mat@ds['vel_rot'].values
            ds['vel_rot'] = xr.DataArray(vel_rot,coords={'coord':['x','y','z'],'time':time},dims=['coord','time'],attrs={'long_name':'IMU Rotational Velocity (PRINC)','units':'m/s','Principle Angle (rad CW of East)':theta})
            vel_trans = rot_mat@ds['vel_trans'].values
            ds['vel_trans'] = xr.DataArray(vel_trans,coords={'coord':['x','y','z'],'time':time},dims=['coord','time'],attrs={'long_name':'IMU HP Translational Velocity (PRINC)','units':'m/s','Principle Angle (rad CW of East)':theta})
    ds.attrs['Coordinate system'] = 'PRINC'
    return ds
    

def PRINC2ENU(ds):
    raise NotImplementedError # I am not sure if I need this OR if it is worth my time in figuring out how to do it!

def ENU2XYZ(ds):
    time = ds['time'].values
    OMAT = ds['orient_mat'].values
    vel_xyz =  np.einsum('ijk,j...k->i...k',OMAT,ds['vel'].values)
    ds['vel'] = xr.DataArray(vel_xyz,coords={'coord':['x','y','z'],'time':time},dims=['coord','time'],attrs={'long_name':'Velocity (XYZ)','units':'m/s'})
    ds.attrs['Coordinate system'] = 'XYZ'
    return ds


def XYZ2BEAM(ds):
    time = ds['time'].values
    beam = np.linalg.inv((ds['trans_mat'].values)) @ ds['vel'].values # Transform to beam
    ds['vel'] = xr.DataArray(beam,coords={'beam':np.arange(1,4),'time':time},dims=['beam','time'],attrs={'long_name':'Velocity (beam)','units':'m/s'})
    ds.attrs['Coordinate system'] = 'BEAM'
    return ds