import xarray as xr
import numpy as np
from scipy.stats import binned_statistic, binned_statistic_2d
from scipy.ndimage import median_filter, generic_filter

from matplotlib.pyplot import subplots

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


def grid_data_1d(da_x, da_y, x_grid, label = None, statistics = ['mean', 'count', 'std']):
    """
    Grid data using scipy.stats.binned_statistic.
    da_x, da_y and x_grid should all be xarray data arrays
    
    Returns an xarray dataset with statistics for grid cells. Default statistics are  mean, 
    count and standard deviation, but this can be altered by input variable statistics.
    If label is given, the variables will be called <label>_mean, <label>_count, <label>_std 
    """     

    if label is None:
        prefix = ""
    else:
        prefix = f"{label}_"
 
    # Compute statistics
    ds_list = []
    for stat in statistics:
        ds_list.append(da_from_binned_statistic_1d(da_x, da_y, x_grid, stat
                                                   ).to_dataset(name = prefix + stat)
                       )

    return xr.merge(ds_list)


def grid_data_2d(x1, x2, f, dims, **kwargs):
    """
    Bin variable f(x1,x2) using scipy.stats.binned_statistics_2d. Binned data is returned as an xarray DataArray

    Parameters:        
        dims : List with name of dimensions, used when creating coordinates for the DataArray
        **kwargs : Passed to scipy.stats.binned_statistics_2d, see doc for details
        
    Example:
        x = np.random.rand(100)-0.5
        y = 2*np.random.rand(100)-1
        f = x**2+y**2
        grid_data_2d(x, y, f, ['x','y'], statistic='median').plot()
    """
    data = binned_statistic_2d(x1, x2, f,**kwargs)
    x1_center = (data.x_edge[:-1] + data.x_edge[1:])/2
    x2_center = (data.y_edge[:-1] + data.y_edge[1:])/2

    return xr.DataArray(data[0], coords = {dims[0] : x1_center, dims[1] : x2_center})

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


def violin_plot(da, dim, ax=None, plot_hist = True, xlabel=None, ylabel=None, hist_kwargs={}, **kwargs):
    """
    Visualize data from a dataarry with violins plots along the given dimension.
    
    Note: Keywords showextrema/showmedians/showmeans are useful for controlling the appearance of the violins
    
    Parameters:
        da  : xarray.DataArray with data
        dim : Dimension to keep (x-axis). All other dimensions will be collapsed
              into a single dimension representing observations.
        plot_hist : True/False. Plot histogram (default True)
        ax  :  If plot_count is True, ax should be a list of 2 axes   
        xlabel : Optional xlabel (default: dim)
        ylabel : Optional y_label (default: none)
        hist_kwargs : (dict) Keyword arguments passed to matplotlib.pyplot.bar 
                      (used for plotting histogram if plot_hist is True)
        **kwargs : Passed to matplotlib.pyplot.violinplot.
        
    Returns (ax, violins) where violins is the handle from matplotlib.pyplot.violinplot.
    """

    if ax is None:
        if plot_hist:            
            fig,ax = subplots(nrows=2, sharex=True, layout='tight', height_ratios=[3,1])
        else:
            fig,ax = subplots()
            
    if xlabel is None:
        xlabel = dim 
        
    # Collapse dimensions
    collapse_dims = []
    for d in da.dims:
        if d != dim:
            collapse_dims.append(d)
    collapsed_data = da.stack({'observations' : collapse_dims}).dropna(dim=dim, how='all')

    # Remove nans and make data to the form violinplot eats
    N = len(collapsed_data[dim])
    data = [[] for i in range(N)] 
    count = [0 for i in range(N)]

    for i in range(N):
        obs      = collapsed_data.isel({dim : i}).values
        data[i]  = obs[~np.isnan(obs)]
        count[i] = len(data[i]) 
    
    ## Plot violins
    if plot_hist:
        ax_violin = ax[0]
    else:
        ax_violin = ax
    
    # Keyword arguments to viloinplot:
    widths = 0.8*np.mean(np.diff(da[dim].values))
    default_kwargs = {'showmedians' : True,
                      'showmeans' : True,
                      'positions' : collapsed_data[dim].values,
                      'widths'    : widths}
    
    # Overwrite with user defined keyword arguments
    kwargs_all = {**default_kwargs, **kwargs}

    violins = ax_violin.violinplot(data, **kwargs_all)
    
    # Make nicer
    if 'cmedians' in violins.keys():
        violins['cmedians'].set_label('median')
        violins['cmedians'].set_color('steelblue')
        
    if 'cmeans' in violins.keys():
        violins['cmeans'].set_label('mean')
        violins['cmeans'].set_color('black')
        violins['cmeans'].set_linestyle('dotted')
        
    if 'cbars' in violins.keys():
        violins['cbars'].set_label('extent')
        violins['cbars'].set_color('silver')
        violins['cbars'].set_linewidth(0.5)
        violins['cbars'].set_zorder(-10)
        violins['cmaxes'].set_visible(False)
        violins['cmins'].set_visible(False)
        
    ax_violin.legend()
       
    ## Plot histogram
    if plot_hist:
        default_hist_kwargs = {'width' : widths,
                               'alpha' : 0.4}
        hist_kwargs_all = {**default_hist_kwargs, **hist_kwargs}
        ax_hist = ax[1]
        ax_hist.bar(collapsed_data[dim].values, count, **hist_kwargs_all)    
    
    ## Label axes
    ax_violin.set_ylabel(ylabel)
    
    if plot_hist:
        ax_hist.set_ylabel('count')
        ax_hist.set_xlabel(xlabel)
    else:
        ax_violin.set_xlabel(xlabel)

    return ax, violins

def apply_median_filter(da, filter_lengths, filter_dims):
    """
    Apply median filter on data array using scipy.nd_image.median_filter
    
    Parameters:
        da : xarray.DataArray with data to be filtered
        filter_length : Size of median filter to be used.
        filter_dim : Corresponding dimensions. The filter will be applied along those dimensions in da 
        
    Returns a new xr.DataArray with filtered data.

    Example:
    apply_median_filter(da, [9], ['time'])
    apply_median_filter(da, [9,21], ['time', 'range'])
    """
    
    # Expand kernel to same dimensions as da
    kernel_size = [1 for dim in da.dims]
    for (i,dim) in enumerate(da.dims):
        for (N, fdim) in zip(filter_lengths, filter_dims):
            if dim == fdim:
                kernel_size[i] = N
    if max(kernel_size) == 1:
        raise ValueError('No dimension to apply filter along!')
    
    # Create DataArray with filtered data
    filtered_da = da.copy()
    filtered_da.data = median_filter(da, kernel_size, mode='reflect')
    
    return filtered_da


def da_filter(da, filter_fun, filter_size, filter_dims, **kwargs):
    """
    Apply filter on data array using scipy.ndimage.generic_filter
    
    Parameters:
        da : xarray.DataArray with data to be filtered
        filter_fun : Function passed to generic_filter (e.g. np.nanmean)
        filter_size : Size of filter along dimensions in filter_dims. ('size' in generic_filter)  (must be a list)
        filter_dim : Dimensions along which filter will be applied (must be a list)
        **kwargs : Keyword arguments passed to generic_filter
        
    Returns a new xr.DataArray with filtered data.

    Example:
    da_filter(da, np.nanmean, [9], ['time'])
    da_filter(da, np.max, [9,21], ['time', 'range'], mode='wrap')
    """
    
    # Expand kernel to same dimensions as da
    kernel_size = [1 for dim in da.dims]
    for (i,dim) in enumerate(da.dims):
        for (N, fdim) in zip(filter_size, filter_dims):
            if dim == fdim:
                kernel_size[i] = N
    if max(kernel_size) == 1:
        raise ValueError('No dimension to apply filter along!')
    
    # Create DataArray with filtered data
    filtered_da = da.copy()
    filtered_da.data = generic_filter(da, filter_fun, size=kernel_size)
    return filtered_da
