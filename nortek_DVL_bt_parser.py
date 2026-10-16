from pandas import DataFrame, concat, to_datetime
from xarray import Dataset
from numpy import unique, array
from struct import unpack, calcsize
from pathlib import Path

def read_dvl_bt(file):
    """
    Parser for reading Nortek DVL bottom track/water track (DF21/DF22) 
    from binary files that lack the required header for opening 
    in for example MIDAS. 

    Returns an xarray.Dataset with bottom/water track in a format inspired by
    mhkit.dolfyn.read().
    """
    def _read():           
        hdr_data = []
        bt_data  = []
        f = open(file, 'rb') 
        while True:
            # read header
            bytes = f.read(hdr_nbytes)
            if len(bytes) == hdr_nbytes:       
                hdr_data.append(unpack(hdr_format, bytes))
                checksum(bytes[:-2], hdr_data[-1][6])
    
                # read bt 
                bytes = f.read(bt_nbytes)
                bt_data.append(unpack(bt_format, bytes))
                checksum(bytes, hdr_data[-1][5])
            else:
                # End of file
                f.close()
                return (hdr_data, bt_data)

    # Read header and bt/wt data from binary file
    hdr_names, hdr_format = struct_params(header)
    hdr_nbytes = calcsize(hdr_format)
    
    bt_names, bt_format = struct_params(bt_hdr)
    bt_nbytes = calcsize(bt_format)

    (hdr_data, bt_data) = _read()

    # Combine header and bt/wt data
    hdr_df = DataFrame(hdr_data, columns = hdr_names)
    bt_df  = DataFrame(bt_data,  columns = bt_names)
    data_raw = concat([hdr_df, bt_df], axis=1) 

    # Check that serial number is unique
    if len(unique(data_raw.serial_number)) != 1:
        raise Exception('Serial number varies within file')
        
    # Make xarray dataset (Note that not all variables are put into the dataset)
    time = timestamp(data_raw)

    data = {
        'time' : ('time', time),
        'beam' : ('beam', [1, 2, 3, 4]),
        'bottom_track' : ('time', data_raw.id == int(b'0x1b',16)),
        'water_track'  : ('time', data_raw.id == int(b'0x1d',16)),
        'c_sound' : ('time', data_raw.soundspeed),
        'error_status' : ('time', data_raw.error_status),
        'status' : ('time', data_raw.status),
        'temp' : ('time', data_raw.temperature),
        'pressure' : ('time', data_raw.pressure * 10),
        'vel' : (('beam', 'time'), array([data_raw.velB1, 
                                          data_raw.velB2,
                                          data_raw.velB3, 
                                          data_raw.velB4])),
        'distance' : (('beam', 'time'), array([data_raw.distanceB1, 
                                               data_raw.distanceB2,
                                               data_raw.distanceB3, 
                                               data_raw.distanceB4])),
        'uncertainty' : (('beam', 'time'), array([data_raw.uncertaintyB1, 
                                                  data_raw.uncertaintyB2,
                                                  data_raw.uncertaintyB3, 
                                                  data_raw.uncertaintyB4])),
    }
    
    attributes = {
        'serial_number' : data_raw.serial_number[0],
        'source_file' : Path(file).name,
        'data_record_definition_version' : unique(data_raw.version),
        'info' : 'See Nortek DVL Operations and Integration manual (DF21 BottomTrack/DF22 WaterTrack) for detailed information and additional available variables.'
    }

    ds = Dataset(data).assign_attrs(attributes)
    
    # Replace invalid values with nan
    for key in value_invalid_estimate.keys():
        remove_invalid_estimates(ds, key, value_invalid_estimate[key])
    
    # Add attributes to variables
    for var in variable_attrs.keys():
        for attr in variable_attrs[var].keys():
            ds[var].attrs[attr] = variable_attrs[var][attr]
            
    return ds
    
def struct_params(list_of_defs):
    names = []
    formats = []
    for itm in list_of_defs:
        names.append(itm[0])
        formats.append(itm[1])
    return names,"<" + "".join(formats)

def checksum(bytes, cs):
    """
    From Nortek DVL operations and integration manual:
    
    The Checksum is defined as a 16-bits unsigned sum of data (16 bits). 
    The sum shall be initialized to the value of 0xB58C before the 
    checksum is calculated.
    """
    cs0 = int("0xb58c",0)
    cs_res = sum(unpack(f"<{len(bytes)//2}H", bytes)) + cs0
    if cs != (cs_res % 65536):
        raise Exception("Checksum failed!")

def remove_invalid_estimates(ds, var, value):
    is_invalid = abs(ds[var] - value) < 1e-5
    ds[var] = ds[var].where(~is_invalid)
    return

def timestamp(data_raw):
    return to_datetime( dict(year = data_raw.year + 1900, 
                             month = data_raw.month + 1, 
                             day = data_raw.day,
                             hour = data_raw.hour,
                             minute = data_raw.minute,
                             second = data_raw.second,
                             ns = data_raw.usec100 * 1e5
                            ))


"""
10 bytes _HeaderData 
See Nortek DVL Operations and Integration for details.
"""
header = [
    ("sync", "B"),
    ("hsz", "B"),
    ("id", "B"),
    ("fam", "B"),
    ("sz", "H"),
    ("cs", "H"),
    ("hcs", "H"),
    ]

"""
DF21 BottomTrack / DF22 WaterTrack 
See Nortek DVL Operations and Integration) for details.
"""
bt_hdr = [
    ("version", "B"),
    ("data_offset", "B"),
    ("serial_number", "I"),
    ("year", "B"),
    ("month", "B"),
    ("day", "B"),
    ("hour", "B"),
    ("minute", "B"),
    ("second", "B"),
    ("usec100", "H"),
    ("n_beams", "H"),
    ("error_status", "I"),
    ("status", "I"),
    ("soundspeed", "f"),
    ("temperature", "f"),
    ("pressure", "f"),
    ("velB1", "f"),
    ("velB2", "f"),
    ("velB3", "f"),
    ("velB4", "f"),
    ("distanceB1", "f"),
    ("distanceB2", "f"),
    ("distanceB3", "f"),
    ("distanceB4", "f"),
    ("uncertaintyB1", "f"),
    ("uncertaintyB2", "f"),
    ("uncertaintyB3", "f"),
    ("uncertaintyB4", "f"),
    ("dt1B1", "f"),
    ("dt1B2", "f"),
    ("dt1B3", "f"),
    ("dt1B4", "f"),
    ("dt2B1", "f"),
    ("dt2B2", "f"),
    ("dt2B3", "f"),
    ("dt2B4", "f"),
    ("time_vel_est_B1", "f"),
    ("time_vel_est_B2", "f"),
    ("time_vel_est_B3", "f"),
    ("time_vel_est_B4", "f"),
    ("velX", "f"),
    ("velY", "f"),
    ("velZ1", "f"),
    ("velZ2", "f"),
    ("uncertaintyX", "f"),
    ("uncertaintyY", "f"),
    ("uncertaintyZ1", "f"),
    ("uncertaintyZ2", "f"),
    ("dt1X", "f"),
    ("dt1Y", "f"),
    ("dt1Z1", "f"),
    ("dt1BZ2", "f"),
    ("dt2X", "f"),
    ("dt2Y", "f"),
    ("dt2Z1", "f"),
    ("dt2Z2", "f"),
    ("time_vel_est_X", "f"),
    ("time_vel_est_Y", "f"),
    ("time_vel_est_Z1", "f"),
    ("time_vel_est_Z2", "f"),
]

value_invalid_estimate = {'vel' : -32.768,
                          'distance' : 0,
                          'uncertainty' : 10
                         }

variable_attrs = {
    'bottom_track' : {'long_name' : 'Is bottom track'},
    'water_track' :  {'long_name' : 'Is water track'},
    'c_sound' : {'units' : 'm s-1',
                 'long_name' : 'Speed of Sound'},
    'error_status' : {'units' : '32 bits',
                      'long_name' : 'Error bit mask',
                      'info' : 'See See Nortek DVL Operations and Integration manual (DF21/DF22 Error status)'},
    'status' : {'units' : '32 bits',
                'long_name' : 'Status bit mask',
                'info' : 'See See Nortek DVL Operations and Integration manual (DF21/DF22 Status)'},
    'temp' : {'units' : 'degree_C',
              'long_name' : 'Temperature'},
    'pressure' : {'units' : 'dbar',
              'long_name' : 'Pressure'},
    'vel' : {'units' : 'm s-1',
             'long_name' : 'Velocity'},
    'distance' : {'units' : 'm',
                  'long_name' : 'Distance',
                  'info' : 'For bottom track: Vertical Distance \nFor water track: Distance along beam to start of water track cell'},
    'uncertainty' : {'units' : 'm s-1',
                     'long_name' : 'Estimated velocity uncertainty',
                     'info' : 'Reported as one standard deviation (Figure of merit - FOM)'}
}
