from glob import glob
from pandas import read_csv, concat, DataFrame
from gsw import SP_from_SA, C_from_SP, t_from_CT, z_from_p, sound_speed, rho
from scipy.stats import binned_statistic
import numpy as np

def CTD_profile_for_depth_conversion(ctdfiles, savefile, bin_size = 1, p_min = 5, SA='AbsSal (g/kg)', CT = 'ConsTemp (deg)', 
                                     p = 'Pressure (dB)', lon = 'Longitude (deg)', lat = 'Latitude (deg)'):
    """
    Creates a csv file with the following columns:
        Depth (m)
        Soundvelocity (m/s)
        Pressure (Bar)
        Density (kg/m3)
        Salinity (pss78)
        Conducitvity (mS/cm)
        In-situ temperature (°C)
    Used for converting pressure to depth in Eivas software using TEOS-10 standard.

    Example:
        >>> CTD_profile_for_depth_conversion('data/CTD/*.txt', 'data/CTD_profile_for_Eiva.csv')

        >>> CTD_profile_for_depth_conversion(['data/CTD/01.txt', 'data/CTD/03.txt'] , 'data/CTD_profile_for_Eiva.csv')

    """

    ctd_data = read_ctd_data(ctdfiles)
    add_columns(ctd_data, SA=SA, CT=CT, p=p, lon=lon, lat=lat)

    # Compute horizontally averaged profiles    
    bins = np.arange(p_min,np.max(ctd_data['Pressure (dB)']),bin_size)  
    (depth,ps) = mean_profile(ctd_data, 'depth', p, bins)
    (c,ps)     = mean_profile(ctd_data, 'c',  p, bins)
    (rho,ps)   = mean_profile(ctd_data, 'rho', p, bins)
    (SP,ps)    = mean_profile(ctd_data, 'SP', p, bins)
    (con,ps)   = mean_profile(ctd_data, 'cond', p, bins)
    (t,ps)     = mean_profile(ctd_data, 't', p, bins)

    # Create new dataframe
    data = DataFrame({'Depth (m)' : depth,
                     'Soundvelocity (m/s)' : c,
                     'Pressure (Bar)' : ps/10,
                     'Density (kg/m3)': rho,
                     'Salinity (pss78)': SP,
                     'Conducitvity (mS/cm)':con,
                     'In-situ temperature (°C)':t
                    })
    
    # Replace nan with lineraly interpolated values
    data = data.interpolate()

    # Save as csv
    data.to_csv(savefile, sep=',', index=False)
    print(f'Saved profile to {savefile}')

    return


def read_ctd_data(ctd_files):
    """
    Load provided csv-files as a pandas.DataFrame
    """
    ctd_data_list = [read_csv(f) for f in list_of_files(ctd_files)]
    return concat(ctd_data_list, axis=0, ignore_index = True)
    

def add_columns(df, SA='AbsSal (g/kg)', CT = 'ConsTemp (deg)', p = 'Pressure (dB)', 
                lon = 'Longitude (deg)', lat = 'Latitude (deg)'):
    df['SP']    = SP_from_SA(df[SA], df[p], df[lon], df[lat])
    df['t']     = t_from_CT(df[SA], df[CT], df[p])
    df['cond']  = C_from_SP(df['SP'], df['t'], df[p])
    df['depth'] = -z_from_p(df[p], df[lat])
    df['c']     = sound_speed(df[SA], df[CT], df[p])
    df['rho']   = rho(df[SA], df[CT], df[p])
    return


def mean_profile(data, var, p, bins):
    """
    Computes mean profile
    """
    res = binned_statistic(data[p].values,
                           data[var].values,
                           statistic='mean', 
                           bins= bins)
    values = res.statistic
    ps = (res.bin_edges[:-1] + res.bin_edges[1:])/2
    return (values, ps)


def list_of_files(filenames):
    """
    Returns a list of files matching the input

    Example usage:
    >>> list_of_files('path/to/files/*.csv')
    ['path/to/files/file1.csv', 'path/to/files/file2.csv']
    
    >>> list_of_files(['path/to/files/*.csv', 'otherfile.txt'])
    ['path/to/files/file1.csv', 'path/to/files/file2.csv', 'otherfile.txt']
    """
    if type(filenames) in (list,tuple):
        file_list = []
        for filename in filenames:
            for f in glob(filename):
                file_list.append(f)
    else:
        file_list = glob(filenames)
    return file_list
