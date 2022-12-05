from struct import unpack
import numpy as np
import xarray as xr
import pandas as pd

def read(path,outpath=None, analog_1_name=None, analog_2_name=None, verbose=False):
    """
    Reads a Nortek .vec file and saves a netCDF to the working 
    
    ----------
    path : str
        Path to Nortek ".vec" file
    outpath: str
        Path to folder to store output. Defaults to current working folder
    analog_1_name : str
        Name of first analog input
    analog_2_name : str
        Name of second analog input 
    verbose : bool
        Words and things
        
    Returns
    -------
    ds: xarray dataset
        Dataset with xrADV labels
    
    """
    # Vector Data Init
    vel_1, vel_2, vel_3 = [],[],[]
    amp_1, amp_2, amp_3 = [],[],[]
    cor_1, cor_2, cor_3 = [],[],[]
    analog_1, analog_2 = [],[]
    pressure = []

    # IMU Data Init
    has_imu = False
    dVel_1, dVel_2, dVel_3 = [],[],[]
    dAng_1, dAng_2, dAng_3 = [],[],[]
    M11,M12,M13,M21,M22,M23,M31,M32,M33 = [],[],[],[],[],[],[],[],[]
    timer = []

    # System Data Init
    time_stamp, battery, sound_speed = [],[],[]
    heading, pitch, roll = [],[],[]
    temperature,error,status,analog_input = [],[],[],[]

    # Output Dict Init
    hardware_properties = {}; head_properties = {}
    user_properties = {}; vel_header_properties = {}
    transMat = []

    ####################################################
    ### Read in file
    with open(path,'rb') as f:
        while True:

####################################################
### Read sync byte and exit if file is finished
            ## Check if you are on a sync byte
            try:
                sync = unpack('<B',f.read(1))[0]
            except: break 

            if (sync == 165):
                try:
                    ID = unpack('<B',f.read(1))[0]
                except: break   
                if ID == 16: # Vector data block does not contain size
                    size = 24
                    try:
                        byts = f.read(size-2)
                    except: break

                    if len(byts) != size-2: print('Reached end of file'); break

                else:
                    try:
                        size = unpack('<H',f.read(2))[0]*2
                        byts = f.read(size-4)
                    except:
                        break
                    if len(byts) != size-4: print('Reached end of file'); break

            else: # Somtimes the sync byte is missing. RIP Chris
                ID = sync ## Assume you read the ID instead of the sync
                if ID == 16: # Vector data block does not contain size
                    try:
                        size =  23 # Hard coded to account for missing sync
                        byts = f.read(size-2)
                    except: break

                    if len(byts) != size-2: print('Reached end of file'); break

                else:
                    try:
                        size = unpack('<H',f.read(2))[0]*2
                        byts = f.read(size-4)
                    except: break

                    if len(byts) != size-4: print('Reached end of file'); break

####################################################
### Check what block you are in and read it.
        ## Hardware Config
            if (ID == 5) & (size == 48):
                hardware_properties = read_hardware_config(byts)
                if verbose: print('Read hardware configuration')
            ## Head Config
            elif (ID == 4) & (size == 224):
                head_properties, transMat = read_head_config(byts)
                if verbose: print('Read head configuration')
            ## User Config
            elif (ID == 0) & (size == 512):
                user_properties = read_user_config(byts)
                if verbose: print('Read user configuration')
            ## Velocity header
            elif (ID == 18) & (size == 42):
                vel_header_properties = read_velocity_header(byts)
                if verbose: print('Read velocity header')

            ## Check data
            elif (ID == 7):
                ## Cant be bothered coding check data - just move to next block (which is annoying for check data)
                None

            ## System 
            elif (ID == 17) & (size == 28):
                sensor_data = read_sensor_data(byts)
                time_stamp.append(sensor_data['time'])
                battery.append(sensor_data['Battery voltage'])
                sound_speed.append(sensor_data['Sound speed'])
                heading.append(sensor_data['Heading'])
                pitch.append(sensor_data['Pitch'])
                roll.append(sensor_data['Roll'])
                temperature.append(sensor_data['Temperature'])
                error.append(sensor_data['Error'])
                status.append(sensor_data['Status'])
                analog_input.append(sensor_data['Analog input'])

            ## Vector data 
            elif (ID == 16):
                vector_data = read_vector_data(byts)
                vel_1.append(vector_data['vel_1'])
                vel_2.append(vector_data['vel_2'])
                vel_3.append(vector_data['vel_3'])
                amp_1.append(vector_data['amp_1'])
                amp_2.append(vector_data['amp_2'])
                amp_3.append(vector_data['amp_3'])
                cor_1.append(vector_data['cor_1'])
                cor_2.append(vector_data['cor_2'])
                cor_3.append(vector_data['cor_3'])
                analog_1.append(vector_data['analog1'])
                analog_2.append(vector_data['analog2'])
                pressure.append(vector_data['pressure']) 

            ## IMU data
            elif (ID == 113) & (size == 72):
                has_imu = True
                imu_data = read_imu_data(byts)
                dVel_1.append(imu_data['dVel x'])
                dVel_2.append(imu_data['dVel y'])
                dVel_3.append(imu_data['dVel z'])
                dAng_1.append(imu_data['dAng x'])
                dAng_2.append(imu_data['dAng y'])
                dAng_3.append(imu_data['dAng z'])
                M11.append(imu_data['M11'])
                M12.append(imu_data['M12'])
                M13.append(imu_data['M13'])
                M21.append(imu_data['M21'])
                M22.append(imu_data['M22'])
                M23.append(imu_data['M23'])
                M31.append(imu_data['M31'])
                M32.append(imu_data['M32'])
                M33.append(imu_data['M33'])
                timer.append(imu_data['Timer'])

    ## Sometimes the vector and IMU blocks are different sizes (ie. we miss the first vector or last IMU block),
    ## so we even them up here so that they fit on the same time dimension      
    if len(vel_1) > len(dVel_1):
        vel_1 = vel_1[:-1]
        vel_2 = vel_2[:-1]
        vel_3 = vel_3[:-1]
        amp_1 = amp_1[:-1]
        amp_2 = amp_2[:-1]
        amp_3 = amp_3[:-1]
        cor_1 = cor_1[:-1]
        cor_2 = cor_2[:-1]
        cor_3 = cor_3[:-1]
        analog_1 = analog_1[:-1]
        analog_2 = analog_2[:-1]
        pressure = pressure[:-1]
    elif len(vel_1) < len(dVel_1):
        dVel_1 = dVel_1[:-1]
        dVel_2 = dVel_2[:-1]
        dVel_3 = dVel_3[:-1]
        dAng_1 = dAng_1[:-1]
        dAng_2 = dAng_2[:-1]
        dAng_3 = dAng_3[:-1]
        M11 = M11[:-1]
        M12 = M12[:-1]
        M13 = M13[:-1]
        M21 = M21[:-1]
        M22 = M22[:-1]
        M23 = M23[:-1]
        M31 = M31[:-1]
        M32 = M32[:-1]
        M33 = M33[:-1]
        timer = timer[:-1]

####################################################
### Initialise xarray object
    ds = xr.Dataset()

####################################################
### Lets make some coordinates i.e. time, time (1hz), matrix rows and columns (i + j),number bit (from rhs)

### Fast time
    if vel_header_properties:
        start_time = vel_header_properties['Time of first measurement']
    else:
        if verbose: print('Velocity header missing, using sensor data time stamp as start time')
        start_time = time_stamp[0]

    ds.attrs['Record length'] = len(vel_1)
    record_length = len(vel_1)

    ## clock is different if the system has an IMU
    if has_imu ==True:
        ave_timestep = round(np.mean(np.diff(np.asarray(timer[:10000])/62500))*1000)
    else:
        ave_timestep = user_properties['Nominal Sample rate [Hz]']*1000
    actual_sampling_rate = round(1000/ave_timestep,4)
    ds.attrs['Sampling Rate [Hz]'] = actual_sampling_rate

    time = np.array(pd.date_range(start=start_time,periods=record_length,freq=str(ave_timestep)+'L'),dtype='datetime64[ns]')

    ds['time'] = xr.DataArray(time,dims='time',attrs={'long_name':'Time'})

    ### Matrix labels
    ds['beam'] = xr.DataArray([1,2,3],dims='beam')
    ds['coord'] = xr.DataArray(['x','y','z'],dims='coord')
    ds['i'] = xr.DataArray([0,1,2],dims='i')
    ds['j'] = xr.DataArray([0,1,2],dims='j')
    ds['bit'] = xr.DataArray([7,6,5,4,3,2,1,0],dims='bit',attrs={'long_name':'Bit number (numbered right-left)'})

    ### Slow time
    sensor_time = np.array(time_stamp,dtype='datetime64[ns]')
    ds['time_slow'] = xr.DataArray(sensor_time,dims='time_slow',attrs={'long_name':'Sensor Time'})

    ### Clear  memory
    for var in [timer,time_stamp,sensor_time,time]:
        del var

####################################################
### Velocity data
    ds['vel'] = xr.DataArray(np.asarray([vel_1,vel_2,vel_3])*0.001*user_properties['Velocity scaling'],\
                             dims=['coord','time'],attrs={'long_name':'Velocity','units':'m/s'})

    ds['amp'] = xr.DataArray(np.asarray([amp_1,amp_2,amp_3]),\
                             dims=['beam','time'],attrs={'long_name':'Beam Amplitude','units':'counts'})

    ds['cor'] = xr.DataArray(np.asarray([cor_1,cor_2,cor_3]),\
                             dims=['beam','time'],attrs={'long_name':'Beam Correlation','units':'%'})

    ds['pressure'] = xr.DataArray(np.asarray(pressure),\
                                  dims=['time'],attrs={'long_name':'Pressure','units':'dbar'})

    ds['pressure'] = xr.DataArray(np.asarray(pressure),\
                                  dims=['time'],attrs={'long_name':'Pressure','units':'dbar'})

    ## Analog channels
    if np.count_nonzero(analog_1) != 0:
        if analog_1_name:
            ds[analog_1_name] = xr.DataArray(np.asarray(analog_1),\
                                          dims=['time'],attrs={'long_name':analog_1_name})
        else:
            ds['analog_1'] = xr.DataArray(np.asarray(analog_1),\
                                          dims=['time'],attrs={'long_name':'Analog 1'})
    else: 
        if verbose: print('Velocity Data: "Analog 1" empty. Not included in output');None

    if np.count_nonzero(analog_2):
        if analog_2_name:
            ds[analog_2_name] = xr.DataArray(np.asarray(analog_2),\
                                          dims=['time'],attrs={'long_name':analog_2_name})
        else:
            ds['analog_2'] = xr.DataArray(np.asarray(analog_2),\
                                          dims=['time'],attrs={'long_name':'Analog 2'})
    else: 
        if verbose: print('Velocity data: "Analog 2" empty. Not included in output')

    ### Clear  memory
    for var in [vel_1,vel_2,vel_3,amp_1,amp_2,amp_3,cor_1,cor_2,cor_3,pressure,analog_1,analog_2]:
        del var
####################################################
### IMU data
    if has_imu == True:
        if verbose: print('IMU Identified')
        
        o_mat = np.array([[M11,M12,M13],[M21,M22,M23],[M31,M32,M33]])
        
        ## Transform orientation matrix from NED to IMU 
        (o_mat[2],o_mat[0]) = (o_mat[0],-o_mat[2].copy())
        o_mat[:,2]*= -1
        (o_mat[:,0],o_mat[:,1]) = (o_mat[:,1],o_mat[:,0].copy())

        ds['orient_mat'] = xr.DataArray(o_mat,\
                                        dims=['i','j','time'],attrs={'long_name':'Orientation Matrix (ENU)'})

        ## Transform accelerometer and gyroscope into scientific units and rotate into ADV frame (from IMU)
        acc = np.asarray([dVel_3,dVel_2,dVel_1])
        acc[0,:] *= -1
        acc *= 9.80665*actual_sampling_rate

        ds['acc'] = xr.DataArray(acc,\
                                 dims=['coord','time'],attrs={'long_name':'Instrument Acceleration','units':'m/s','Coordinate system':'XYZ'})

        omega = np.asarray([dAng_3,dAng_2,dAng_1])*actual_sampling_rate
        omega[0,:] *= -1

        ds['omega'] = xr.DataArray(omega,\
                                   dims=['coord','time'],attrs={'long_name':'Instrument Angular rate','units':'rad/s','Coordinate system':'XYZ'})

        for var in [M11,M12,M13,M21,M22,M23,M31,M32,M33,dVel_1,dVel_2,dVel_3,dAng_1,dAng_2,dAng_3,acc,omega]:
            del var

####################################################
### Transformation Matrix
    ds['trans_mat'] = xr.DataArray(transMat,\
                                   dims = ['i','j'],attrs={'long_name':'XYZ-Beam transformation matrix'})

####################################################
### Sensor Data
    ds['battery'] = xr.DataArray(np.asarray(battery)*0.1,\
                                 dims=['time_slow'],attrs={'long_name':'Battery Voltage','units':'Volts'})
    ds['sound_speed'] = xr.DataArray(np.asarray(sound_speed)*0.1,\
                                     dims=['time_slow'],attrs={'long_name':'Speed of sound','units':'m/s'})
    ds['temperature_sensor'] = xr.DataArray(np.asarray(temperature)*0.01,\
                                     dims=['time_slow'],attrs={'long_name':'Sensor Temperature','units':'Degrees C'})
    ds['error'] = xr.DataArray(np.asarray(error),\
                               dims=['time_slow','bit'],attrs={'long_name':'Error Code'})
    ds['status'] = xr.DataArray(np.asarray(status),\
                                dims=['time_slow','bit'],attrs={'long_name':'Status Code'})
## If analog signal is empty or populated only with zeros, dont include it
    if np.count_nonzero(analog_input) != 0:
        ds['sensor_analog'] = xr.DataArray(np.asarray(analog_input),\
                                           dims=['time_slow'],attrs={'long_name':'Sensor analog input'})
    else:
        if verbose: print('Sensor data: "analog input" empty. Not included in output')
## Same with pitch,roll + heading data
    if np.count_nonzero(heading) != 0:
        ds['heading'] = xr.DataArray(np.asarray(heading)*0.1,\
                                     dims=['time_slow'],attrs={'long_name':'Heading'})
    else:
        if verbose: print('Sensor data: "heading" empty. Not included in output')

    if np.count_nonzero(pitch) != 0:
        ds['pitch'] = xr.DataArray(np.asarray(pitch)*0.1,\
                                   dims=['time_slow'],attrs={'long_name':'Pitch','units':'Degrees'})
    else:
        if verbose: print('Sensor data: "pitch" empty. Not included in output')

    if np.count_nonzero(roll) != 0:
        ds['roll'] = xr.DataArray(np.asarray(roll)*0.1,\
                                  dims=['time_slow'],attrs={'long_name':'Roll','units':'Degrees'})
    else:
        if verbose: print('Sensor data: "roll" empty. Not included in output')

    ### Clear  memory
    for var in [analog_input,heading,pitch,roll,battery,sound_speed,temperature,error,status]:
        del var

####################################################
### Properties
### Add properties to the netCDF

    # I will order these by things I am interested in.
    ds.attrs['Coordinate system'] = user_properties['Coordinate System']
    ds.attrs['Hardware serial number'] = hardware_properties['Hardware serial number']
    ds.attrs['Head serial number'] = head_properties['Head serial number'] # bug here maybe
    ds.attrs['Firmware version'] = hardware_properties['Firmware Version']
    ds.attrs['Number of beams'] = head_properties['Number of beams']
    ds.attrs['Head type'] = head_properties['Head type'] 
    ds.attrs['Head frequency'] = head_properties['Head frequency [kHz]']
    ds.attrs['Board config'] = hardware_properties['Board config']
    ds.attrs['Board freq [kHz]'] = hardware_properties['Board freq [kHz]']
    ds.attrs['ProLog ID'] = hardware_properties['ProLog ID']
    ds.attrs['ProLog firmware'] = hardware_properties['ProLog firmware']
    ds.attrs['Hardware revision'] = hardware_properties['Hardware revision']

    if vel_header_properties:
        ds.attrs['Noise1'] = vel_header_properties['Noise1']
        ds.attrs['Noise2'] = vel_header_properties['Noise2']
        ds.attrs['Noise3'] = vel_header_properties['Noise3']
        ds.attrs['Correlation1'] = vel_header_properties['Correlation1']
        ds.attrs['Correlation2'] = vel_header_properties['Correlation2']
        ds.attrs['Correlation3'] = vel_header_properties['Correlation3']

    else:
        if verbose: print('Velocity header missing, not added to attrs')
        start_time = time_stamp[0]


    fname = path.split(sep='\\')[-1].split('.')[0]
    if outpath:
        ds.to_netcdf(path=outpath)
    else:
        ds.to_netcdf(path=fname+'.nc')


#########################################################################
#########################################################################
### Here are the readers for each block and some general helper functions 
### that I got from Kilchers Dolfyn package. Each function is passed the
### bytes from the f.read (not including sync, ID and size) and then these
### are unpacked into temporary array which is in turn parsed into dicts 
### and returned

def read_hardware_config(byts):
    properties_dict = dict()
    temp = unpack('<14s6H12x4sH',byts)
    
    properties_dict['Hardware serial number'] = temp[0][:8].decode('utf-8')
    properties_dict['ProLog ID'] = unpack('B',temp[0][8:9])[0]
    properties_dict['ProLog firmware'] = temp[0][10:].decode('utf-8')
    properties_dict['Board config'] = temp[1]
    properties_dict['Board freq [kHz]'] = temp[2]
    properties_dict['PIC version'] = temp[3]
    properties_dict['Hardware revision'] = temp[4]
    properties_dict['Recorder size [MB]'] = temp[5]*65536*2**-20
    properties_dict['Status'] = temp[6]
    properties_dict['Firmware Version'] = temp[7].decode('utf-8')
    
    return properties_dict

def read_head_config(byts):
    properties_dict = dict()
    temp = unpack('<3H12s176s22s2H',byts)
    
    properties_dict['Pressure sensor'] = bool(bin(temp[0])[2][0])
    properties_dict['Magnetometer sensor'] = bool(bin(temp[0])[3][0])
    properties_dict['Tilt sensor'] = bool(bin(temp[0])[4][0])
    
    properties_dict['Head frequency [kHz]'] = temp[1]
    properties_dict['Head type'] = temp[2]
    properties_dict['Head serial number'] = temp[3].decode('utf-8')
    properties_dict['System'] = temp[4]
    properties_dict['Number of beams'] = temp[6]
    
    transMat = np.array(unpack('<9h', temp[4][8:26])).reshape(3, 3) / 4096.

    return properties_dict , transMat

def read_user_config(byts):
    properties_dict = dict()
    temp = unpack('<18H6s4HI8H2x90H180s6H4xH2x2H2xH30x9H',byts)
    
    properties_dict['Transmit length [counts]'] = temp[0]
    properties_dict['Blanking distance [counts]'] = temp[1]
    properties_dict['Receive length [counts]'] = temp[2]
    properties_dict['Time between pings [counts]'] = temp[3]
    properties_dict['Time between bursts [counts]'] = temp[4]
    properties_dict['Npings'] = temp[5]
    properties_dict['Nominal Sample rate [Hz]'] = 512/temp[6]
    properties_dict['Nbeams'] = temp[7]
    
    properties_dict['Coordinate System'] = ['ENU','XYZ','BEAM'][temp[14]]
    
    tbyts = byts[44:50]
    properties_dict['Deployment time'] = read_time(tbyts)
     
    
    properties_dict['Deployment name'] = temp[18].partition(b'\x00')[0].decode('utf-8')
    
    mode0 =  int2binarray(temp[24], 16)
    properties_dict['Velocity scaling'] = [1, 0.1][mode0[4].astype('int')]
    # Can add the others eventually: properties_dict[''] = temp[] 
    return properties_dict

def read_velocity_header(byts):
    properties_dict = dict()
    temp = unpack('<6xH7B21xH',byts)
    
    properties_dict['NRecords'] = temp[0]
    properties_dict['Noise1'] = temp[1]
    properties_dict['Noise2'] = temp[2]
    properties_dict['Noise3'] = temp[3]
    properties_dict['Spare0'] = temp[4]
    properties_dict['Correlation1'] = temp[5]
    properties_dict['Correlation2'] = temp[6]
    properties_dict['Correlation3'] = temp[7]
    properties_dict['Spare1'] = temp[8]
    
    tbyts = byts[0:6]
    properties_dict['Time of first measurement'] = read_time(tbyts)
    # Can add the others eventually: properties_dict[''] = temp[]    
    return properties_dict

def read_vector_data(byts):
    data_dict = dict()
    temp = unpack('<4B2H3h6BH',byts)
    
    data_dict['analog1'] = temp[5]
    data_dict['analog2'] = temp[0] + temp[3] ## IDK this
    data_dict['pressure'] = (65536*temp[2] + temp[4])/1000

    data_dict['vel_1'] = temp[6]
    data_dict['vel_2'] = temp[7]
    data_dict['vel_3'] = temp[8]
    
    data_dict['amp_1'] = temp[9]
    data_dict['amp_2'] = temp[10]
    data_dict['amp_3'] = temp[11]
    
    data_dict['cor_1'] = temp[12]
    data_dict['cor_2'] = temp[13]
    data_dict['cor_3'] = temp[14]
    
    return data_dict

def read_sensor_data(byts):
    properties_dict = dict()
    
    temp = unpack('<6B2H3hH2B2H',byts)
    
    properties_dict['time'] = read_time(byts[0:6])
    properties_dict['Battery voltage'] = temp[6]
    properties_dict['Sound speed'] = temp[7]
    properties_dict['Heading'] = temp[8]
    properties_dict['Pitch'] = temp[9]
    properties_dict['Roll'] = temp[10]
    properties_dict['Temperature'] = temp[11] 
    properties_dict['Error'] = int2binarray(temp[12],8).astype('int')[::-1]
    properties_dict['Status'] = int2binarray(temp[13],8).astype('int')[::-1]
    properties_dict['Analog input'] = temp[14]
    
    return properties_dict

def read_imu_data(byts):
    data_dict = dict()
    temp = unpack('<BB15fiH',byts)
    
    data_dict['Counter'] = temp[0]
    data_dict['AHRS ID'] = temp[1]
    data_dict['dAng x'] = temp[2]
    data_dict['dAng y'] = temp[3]
    data_dict['dAng z'] = temp[4]
    data_dict['dVel x'] = temp[5]
    data_dict['dVel y'] = temp[6]
    data_dict['dVel z'] = temp[7]
    data_dict['M11'] = temp[8]
    data_dict['M12'] = temp[9]
    data_dict['M13'] = temp[10]
    data_dict['M21'] = temp[11]
    data_dict['M22'] = temp[12]
    data_dict['M23'] = temp[13]
    data_dict['M31'] = temp[14]
    data_dict['M32'] = temp[15]
    data_dict['M33'] = temp[16]
    data_dict['Timer'] = temp[17]
    return data_dict

## Helper functions
def int2binarray(val, n):
    out = np.zeros(n, dtype='bool')
    for idx, n in enumerate(range(n)):
        out[idx] = val & (2 ** n)
    return out

def _bcd2char(cBCD):
    """
    Taken from the Nortek System Integrator
    Manual "Example Program" Chapter.
    """
    cBCD = min(cBCD, 153)
    c = (cBCD & 15)
    c += 10 * (cBCD >> 4)
    return c

def read_time(tbyts):
    time_bcd = unpack('BBBBBB',tbyts)
    
    time_int = [str(_bcd2char(x)) for x in time_bcd]
    time_int = ["0"+tint if len(tint) == 1 else tint for tint in time_int]
    
    time_str = '20'+time_int[4]+'-'+time_int[5]+'-'+time_int[2]+' '+time_int[3]+':'+time_int[0]+':'+time_int[1]
    return time_str