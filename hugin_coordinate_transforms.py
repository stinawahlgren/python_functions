from xarray import DataArray, dot, apply_ufunc
from numpy import array, diag
from scipy.spatial.transform import Rotation

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
    return dot(T_rot, T_flip, dim='auv_ps_flipped')
