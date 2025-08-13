import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def read_hugin_nav_postprocessed(mission_folder, version, suffix=None):
    """
    Reads position and attitude from postprocessed files 
    
    version = 'NBP22': data is taken from attitude_smooth.txt 
                       and position_smooth.txt in {mission_folder}/post/
    """

    if version == 'NBP22':
        # Read attitude data
        if suffix is None:
            suffix = ''
        attitude = pd.read_csv(Path(mission_folder).joinpath(f'post/attitude_smooth{suffix}.txt'),
                               sep = '\s+',
                               usecols = [0,1,2,3],
                               names = ['time', 'roll', 'pitch', 'heading' ])
    
        attitude['time'] = pd.to_datetime(attitude['time'], unit='s')
    
        # Read position data
        position = pd.read_csv(Path(mission_folder).joinpath(f'post/position_smooth{suffix}.txt'),
                               sep = '\s+',
                               usecols = [0,1,2,3],
                               names = ['time', 'latitude', 'longitude', 'depth' ])
        position['time'] = pd.to_datetime(position['time'], unit='s')
    
        # Make sure that files agree
        assert_data_matching([attitude, position])
    
        # Combine to one dataframe
        nav = pd.concat([attitude,
                         position[['latitude', 'longitude', 'depth']]],
                        axis=1)
    else:
        raise ValueError("Version not supported. Supported versions are: 'NBP22'")

    # Convert from radians to degrees
    for col in ['roll', 'pitch', 'heading', 'latitude', 'longitude']:
        nav[col] = nav[col]*180/np.pi

    return nav

def read_hugin_nav(mission_folder):
    """
    Combine mutliple nav files from Ran to a pandas dataframe
    """
    
    df_list = []
    
    try:
        df_list.append(read_navpos(mission_folder))
    except:
        print('Omitting navpos.txt (not found in mission folder)')
    
    try:
        df_list.append(read_navhead(mission_folder))
    except:
        print('Omitting navhead.txt (not found in mission folder)')
    
    try:
        df_list.append(read_vel(mission_folder))
    except:
        print('Omitting vel.txt (not found in mission folder)')
        
    try:
        df_list.append(read_head(mission_folder))
    except:
        print('Omitting head.txt (not found in mission folder)')
    
    try:
        df_list.append(read_depth(mission_folder))
    except:
        print('Omitting depth.txt (not found in mission folder)')
        
    try:
        df_list.append(read_sensstat(mission_folder))
    except:
        print('Omitting sensstat.txt (not found in mission folder)')
        
    assert_data_matching(df_list)
                    
    df = pd.concat(df_list, axis=1)
    
    # Remove duplicate columns
    return df.loc[:, ~df.columns.duplicated()]

def read_navpos(mission_folder):
    """
    Reads time, longitude, latitude and depth from navpos.txt in the given mission folder.
    
    Note: All column headers are listed in format.txt
    """
    
    file = Path(mission_folder).joinpath('cp/Data/navpos.txt')

    df = pd.read_csv(file,
                     sep = '\t',
                     usecols = [0, 7, 8, 9, 11,12,15],
                     names = ['time', 
                              'DR_NAV_VEL_NORTH',
                              'DR_NAV_VEL_EAST',
                              'DR_NAV_VEL_FORWARD',
                              'NAV_LATITUDE', 
                              'NAV_LONGITUDE',
                              'NAV_DEPTH'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df
            
def read_head(mission_folder):
    """
    Reads primary roll, pitch from head.txt in the given mission folder.
    """
    
    file = Path(mission_folder).joinpath('cp/Data/head.txt')

    df = pd.read_csv(file,
                     sep = '\t',
                     usecols = [0, 2, 3],
                     names = ['time', 
                              'HD_ROLL',
                              'HD_PITCH'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df



def read_depth(mission_folder):
    """
    Reads primary and secondary altitude and pitch from motion data from depth.txt in the given mission folder.
    """
    
    file = Path(mission_folder).joinpath('cp/Data/depth.txt')

    df = pd.read_csv(file,
                     sep = '\t',
                     usecols = [0, 3, 4,11],
                     names = ['time', 
                              'ALT_ALTITUDE_primary',
                              'ALT_ALTITUDE_secondary',
                              'MD_PITCH'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def read_sensstat(mission_folder):
    """
    Reads FLS status data from sensstat.txt in the given mission folder.
    """
    
    file = Path(mission_folder).joinpath('cp/Data/sensstat.txt')

    df = pd.read_csv(file,
                     sep = '\t',
                     usecols = [0, 43, 44],
                     names = ['time', 
                              'CA_FLS_DRIVER_STATUS',
                              'CA_FLS_BTM_TRK_QLTY'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def read_navhead(mission_folder):
    """
    Reads time, longitude, latitude and depth from navhead.txt in the given mission folder.
    """
    file = Path(mission_folder).joinpath('cp/Data/navhead.txt')

    df = pd.read_csv(file,
                     sep = '\t',
                     usecols = [0,1],
                     names = ['time', 
                              'DR_COMP_HEADING'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def read_vel(mission_folder):
    """
    Reads time, DVL bottom and top track from vel.txt in the given mission folder.
    
    Note that 0s in the velocity measurements are replaced with NaN, 
    as the velocities seem to be reported as 0 when no track is found.
      
    """
    file = Path(mission_folder).joinpath('cp/Data/vel.txt')

    df = pd.read_csv(file,
                     sep = '\t',
                     usecols = [i for i in range(19)] + [21],
                     names = ['time',
                              'DVL_FA_BOTM_VEL',
                              'DVL_PS_BOTM_VEL',
                              'DVL_VERT_BOTM_VEL',
                              'DVL_ERR_BOTM_VEL',
                              'DVL_FA_WTR_VEL',
                              'DVL_PS_WTR_VEL',
                              'DVL_VERT_WTR_VEL',
                              'DVL_ERR_WTR_VEL',
                              'DVL_REF_LAY_START',
                              'DVL_REF_LAY_END',
                              'DVL_BM1_BOTM_TRK_QLTY',
                              'DVL_BM2_BOTM_TRK_QLTY',
                              'DVL_BM3_BOTM_TRK_QLTY',
                              'DVL_BM4_BOTM_TRK_QLTY',
                              'DVL_BM1_WTR_TRK_QLTY',
                              'DVL_BM2_WTR_TRK_QLTY',
                              'DVL_BM3_WTR_TRK_QLTY',
                              'DVL_BM4_WTR_TRK_QLTY',
                              'DVL_WTR_TRACK_MODE'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Replace 0 velocity with NaN
    vel_columns = ['DVL_FA_BOTM_VEL',
                   'DVL_PS_BOTM_VEL',
                   'DVL_VERT_BOTM_VEL',
                   #'DVL_ERR_BOTM_VEL', #Error velocity seems to be zero everywhere?
                   'DVL_FA_WTR_VEL',
                   'DVL_PS_WTR_VEL',
                   'DVL_VERT_WTR_VEL',
                   #'DVL_ERR_WTR_VEL',
                   ]

    for col in vel_columns:
        df.loc[abs(df[col])<1e-9, col] = np.nan
    
    return df

def assert_data_matching(df_list):
    """
    Check that the given list of data frames come from the same mission by assuring
     1. Same number of rows
     2. Time stamps agree within 1 ms
    
    Will raise an error if this is not the case.
    """
    
    if len(df_list) > 1:      
        for df in df_list[1:]:         
            if len(df) != len(df_list[0].index):
                raise ValueError('Different number of rows')
            
            if max(df.time-df_list[0].time) > pd.Timedelta('0.001s'):
                raise ValueError('Timestamps do not match')

def add_AUV_nav(ADCP_ds, AUV_nav, debug_plots = False):
    """
    Returns a new xarray dataset with interpolated data from AUV_nav
    
    Parameters:
        ADCP_ds : Xarray dataset with Pinnacle data
        AUV_nav : Pandas dataframe with navigation data from Ran
        debug_plots : Optional argument. If True, a lot of plots 
            for following the steps will be generated.
    """
    # Convert timestamps to integers for easier interpolation
    # ADCP_time is converted to 1D
    time_variable = 'time'
    
    ADCP_shape = ADCP_ds[time_variable].shape
    ADCP_dims  = ADCP_ds[time_variable].dims
    ADCP_time_ns = ADCP_ds[time_variable].astype(int).values.reshape(-1, order='F')
    AUV_time_ns = AUV_nav.time.astype(int).values
    
    # Make sure that time overlaps
    if max(min(ADCP_time_ns),min(AUV_time_ns)) > min(max(ADCP_time_ns),max(AUV_time_ns)):
        raise ValueError('ADCP and AUV from different missions')
    
    if debug_plots:
        print('----------------------------------------------------------------')
        print('Debug plots from add_AUV_nav:')
        print('----------------------------------------------------------------')
        plt.plot(AUV_time_ns, 0*AUV_time_ns, 'k', label = 'AUV time')
        plt.plot(ADCP_time_ns, 0*ADCP_time_ns, 'y-.', label = 'ADCP time')
        plt.legend()
        plt.title('Make sure time is converted to int in the same way')
        plt.show()

        plt.plot(ADCP_time_ns)
        plt.xlabel('index')
        plt.ylabel('ADCP time')
        plt.title('Make sure ADCP time is reshaped from 2D to increasing 1D')
        plt.show()
        
    # Interpolate AUV_nav onto ADCP_time_int
    interpolated_data = {}
    for col in AUV_nav.columns[AUV_nav.columns!='time']:
        interpolated_data[col] = np.interp(ADCP_time_ns, # new time stamps
                                           AUV_time_ns, # original time stamps
                                           AUV_nav[col].values, # original values
                                           left = np.nan, 
                                           right = np.nan)

    if debug_plots:
        print('----------------------------------------------------------------')
        print('Interpolated variables:')
        print('----------------------------------------------------------------')
        for col in AUV_nav.columns[AUV_nav.columns!='time']:
            plt.plot(AUV_time_ns, AUV_nav[col], 'k',  label ='original')
            plt.plot(ADCP_time_ns, interpolated_data[col], 'y-.', label ='interpolated')
            plt.legend(loc = 'lower left')
            plt.title(col)
            plt.show()
            
    # Add new variables to ADCP dataset
    new_variables = {}
    for key in interpolated_data.keys():
        new_variables[key] = (ADCP_dims, interpolated_data[key].reshape(ADCP_shape,order='F'))

    ADCP_ds = ADCP_ds.assign(new_variables)

    if debug_plots:
        # Check if new variables look reasonable
        
        print('----------------------------------------------------------------')
        print('New varaibles in dataset:')
        print('----------------------------------------------------------------')
        for key in interpolated_data.keys():
            ADCP_ds[key].plot()
            plt.show()
            
    return ADCP_ds
