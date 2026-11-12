from xarray import DataArray, dot, apply_ufunc
from numpy import array, diag
from scipy.spatial.transform import Rotation

"""
Calibration values from calib.ini
"""
dvl_up_calib_Ran = {'roll' : -0.1492,
                    'pitch': -0.3491,
                    'yaw'  : -0.4857}

def auv2enu(ds, heading_var='auv_heading', pitch_var='auv_pitch', roll_var='auv_roll'):
    '''
    Computes the transformation matrix to go from the AUV frame ('fa', 'ps', 'vert'),
    to the earth coordinate frame ('E', 'N', 'U').
    
    The auv frame coordinate system is left-handed (forward, starboard, up), so we 
    first flip the ps-axis. Since pitch is defined relative to the ps-axis, this also 
    means we have to reverse the sign of the pitch.

    Secondly, we have to convert the heading, so that 0 is along the E-axis and so that it 
    is counter clockwise instead of clockwise.

    Those steps have been verified with water velocity data. However, the roll-part have
    not been verified, due to a lack of reference data, so I don't know if that is
    correctly handled...
    '''
    def _rotation_matrix(yaw, pitch, roll):
        return Rotation.from_euler('ZYX',
                                   array([yaw,pitch,roll]).T, 
                                   degrees=True
                                  ).as_matrix()

    T_flip = DataArray(diag([1,-1,1]), dims=['auv','auv_ps_flipped'])
    T_rot  = apply_ufunc(
        _rotation_matrix,
        -(ds[heading_var]-90),
        -ds[pitch_var],
        ds[roll_var],
        input_core_dims=[['time'], ['time'], ['time']],
        output_core_dims=[['time', 'earth', 'auv_ps_flipped']]
    )
    return dot(T_rot, T_flip, dim='auv_ps_flipped'
               ).assign_coords({'earth' : ['E', 'N', 'U'],
                                'auv'   : ['fa', 'ps', 'vert']})


def inst_adcp2inst_dvl_up():
    """
    Transformation matrix from the DVL ADCP coordinate system to the DVL 
    bottom track coordinate system. This is a 135 degree rotation around 
    the z-axis followed by flipping the direction of the y- and z-axis.
    """
    return  DataArray(diag([1,-1,-1]) @ Rotation.from_euler('z',-135, degrees=True).as_matrix(), 
                         coords = {'dvl': ['X', 'Y', 'Z'], 
                                   'adcp' : ['X', 'Y', 'Z']}
                        ).rename('inst_adcp2inst_dvl')


def inst_dvl_up2auv(expedition = 'ANA14B'):

    if expedition in ['NBP22', 'ANA14B']:
        calib = dvl_up_calib_Ran

    return  DataArray(R_calib(calib) @ diag([1,-1,-1]),
                          coords = {'auv' : ['fa', 'ps', 'vert'],
                                    'dvl' : ['X', 'Y', 'Z']}
                         ).rename('inst_dvl2auv')

def inst_adcp2auv(expedition = 'ANA14B'):
    
    if expedition in ['NBP22', 'ANA14B']:
        calib = dvl_up_calib_Ran
        
    return DataArray(R_calib(calib) @ Rotation.from_euler('z',-135, degrees=True).as_matrix(),
                        coords = {'auv' : ['fa', 'ps', 'vert'],
                                  'adcp' : ['X', 'Y', 'Z']}
                        ).rename('inst_adcp2auv')

def R_calib(calib):
    return Rotation.from_euler('zyx',
                               [calib['roll'],
                                calib['pitch'],
                                calib['yaw']],
                               degrees=True).as_matrix()
