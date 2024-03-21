import xarray as xr
import numpy as np
from scipy.stats import binned_statistic

from .misc import get_edges

def has_same_coords(da1, da2):
    """
    Check if two xr.DataArrays have the same coordinates (name and values). 
    The coordinates do not have to come in the same order.
    """
    output = True
    
    # Check if they have the same dimensions
    if set(da1.dims)!=set(da2.dims):
        output = False
     
    # Check if coordinates have same values
    else:
        try:
            xr.align(da1, da2, join='exact')
        except ValueError:
            output = False
            
    return output


def grid_data_1d(da_x, da_y, x_grid, label = None):
    """
    Grid data using scipy.stats.binned_statistic.
    da_x, da_y and x_grid should all be xarray data arrays
    
    Returns an xarray dataset with mean, count and standard deviation for grid cells. 
    If label is given, the variables will be called <label>_mean, <label>_count, <label>_std 
    """     

    if label is None:
        prefix = ""
    else:
        prefix = f"{label}_"
 
    # Compute statistics   
    mean  = da_from_binned_statistic_1d(da_x, da_y, x_grid, 'mean'
                                        ).to_dataset(name = prefix + 'mean')
    count = da_from_binned_statistic_1d(da_x, da_y, x_grid, 'count'
                                        ).to_dataset(name = prefix + 'count')
    std   = da_from_binned_statistic_1d(da_x, da_y, x_grid, 'std'
                                        ).to_dataset(name = prefix + 'std')

    return mean.merge(std).merge(count)


def da_from_binned_statistic_1d(da_x, da_y, x_grid, statistic):
    """
    Parameters:
        da_x : An xarray DataArray with values to be binned
        da_y : An xarray DataArray with the data on which the statistic will be computed.
               Can have one more dimension than da_x, but the common dimensions must have
               same names and coordinate values as da_x.
        x_grid : one-dimensional xarray DataArray with center points of the grid
        statistic : passed to scipy.stats.binned_statistic (e.g 'mean', 'count', 'std')
        
    Returns:
        An xarray DataArray with statistic of the binned data
    """

    different_dimensions = set(da_y.dims) - set(da_x.dims)
    
    grid_dim = list(x_grid.dims)[0]
    
    if len(different_dimensions) == 0:
        dims   = grid_dim
        coords = {grid_dim: x_grid.values}      
        data   = get_binned_statistics(da_x, da_y, x_grid, statistic).statistic
        
    elif len(different_dimensions) == 1:
        keep_dim = list(different_dimensions)[0]
        dims   = (grid_dim, keep_dim)
        coords = {grid_dim: x_grid.values,
                  keep_dim: da_y[keep_dim].values}
        
        len_keep_dim = len(da_y[keep_dim].values) 
        data = np.empty([len(x_grid.values), len_keep_dim])
        for i in range(len_keep_dim):
            data[:,i] = get_binned_statistics(da_x, da_y.isel({keep_dim : i}), x_grid, statistic).statistic
    
    else:
        raise ValueError("da_y can have at most one more dimension than da_x")
            
    # Create DataArray
    da = xr.DataArray(data, 
                      dims   = dims,
                      coords = coords)
    return da


def get_binned_statistics(da_x, da_y, x_grid, statistic):
    """
    Parameters:
        da_x : An xarray DataArray with values to be binned
        da_y : An xarray DataArray with the data on which the statistic will be computed.
               Must have same dimensions and coordinate values as da_x
        x_grid : one-dimensional xarray DataArray with center points of the grid
        statistic : passed to scipy.stats.binned_statistic (e.g 'mean', 'count', 'std')
        
    Returns:
        Output directly from scipy.stats.binned_statistic
    """
    # Make sure grid is 1D:    
    if len(x_grid.dims) != 1:
        raise ValueError('Grid must be one-dimensional!')
        
    # Make sure that x and y have same coordinates
    if not has_same_coords(da_x, da_y):
        raise ValueError("da_x and da_y don't match")
        
    # Make 1D
    dims = da_x.dims
    x = da_x.stack(z=dims).values
    y = da_y.stack(z=dims).values
    
    # Remove Nan
    is_nan = np.isnan(y) | np.isnan(x)
    x = x[~is_nan]
    y = y[~is_nan]
    
    # Compute statistics
    bin_edges = get_edges(x_grid.values)  
    return binned_statistic(x, y, statistic = statistic, bins = bin_edges)
