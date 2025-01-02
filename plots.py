import xarray as xr
import numpy as np
from scipy import stats    #Used for 2D binned statistics

import matplotlib.pyplot as plt
from matplotlib import colors
import cmocean.cm as cmo
from matplotlib.dates import DateFormatter, DayLocator, HourLocator, date2num
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .misc import get_edges

def pcolormesh_offset(x,y,z,y_offset, x_pixel_scale = None, vmin = None, vmax = None, ax = None,  **kwargs):
    """
    Example:
        Ny = 300
        Nx = 400
        x  = np.arange(Nx)
        y  = np.arange(Ny)
        z  = np.random.rand(Ny, Nx)
        y_offset = np.linspace(0,50,Nx)
        pcolormesh_offset(x, y, z, y_offset)
    """    
    (Ny, Nx) = z.shape
    
    if Ny != len(y):
        raise ValueError('y must have same length as first dimension of z')
        
    if Nx != len(x):
        raise ValueError('x must have same length as second dimension of z')
        
    if len(y_offset) != Nx:
        raise ValueError('y_offset must have same length as second dimension of z') 
        
    if vmin is None:
        vmin = np.nanmin(z)
    
    if vmax is None:
        vmax = np.nanmax(z)

    if ax is None:
        ax = plt.gca()
        
    # Deal with nan in offset
    nan_offset = np.isnan(y_offset)   
    y_offset[nan_offset] = 0
    z[:,nan_offset] = np.nan
        
    x_edges = get_edges(x)
    y_edges = get_edges(y)
                             
    for i in range(Nx):
        if x_pixel_scale is None:
            x_edge = x_edges[i:(i+2)]
        else:
            x_center = (x_edges[i]+x_edges[i+1])/2
            x_edge   = np.array([-0.5,0.5])* x_pixel_scale * x_step + x_center
            
        im = ax.pcolormesh(x_edge, y_offset[i]+y_edges, z[:,i:i+1], vmin=vmin, vmax=vmax, **kwargs)
      
    ax.set_xlim(x_edges[0], x_edges[-1])
    return im

def pcolormesh_nongridded_y(x,y,z, x_pixel_scale = None, vmin = None, vmax = None, **kwargs):
    """
    Example:
        Ny = 15
        Nx = 10
        x  = np.arange(Nx)
        y  = np.sort(np.random.rand(Ny, Nx), axis=0)
        z  = np.random.rand(Ny, Nx)
        pcolormesh_nongridded_y(x, y, z)
    """    
    (Ny, Nx) = z.shape
    
    if y.shape != z.shape:
        raise ValueError('y and z must have same size')
        
    if Nx != len(x):
        raise ValueError('x must have same length as second dimension of z')       
        
    if vmin is None:
        vmin = np.nanmin(z)
    
    if vmax is None:
        vmax = np.nanmax(z)
        
    x_edges = get_edges(x)
                             
    for i in range(Nx):
        if x_pixel_scale is None:
            x_edge = x_edges[i:(i+2)]
        else:
            x_center = (x_edges[i]+x_edges[i+1])/2
            x_edge   = np.array([-0.5,0.5])* x_pixel_scale * x_step + x_center
        
        y_edges = get_edges(y[:,i])

        plt.pcolormesh(x_edge, y_edges, z[:,i:i+1], vmin=vmin, vmax=vmax, **kwargs)
      
    plt.xlim(x_edges[0], x_edges[-1])
    

def nice_time_axis(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_minor_locator(HourLocator([0,3,6,9,12,15,18,21]))
    ax.xaxis.set_major_formatter(DateFormatter("%Y %b %d"))
    ax.xaxis.set_minor_formatter(DateFormatter("%H:%M"))
    ax.get_xaxis().set_tick_params(which='major', pad=10)

def fancy_2d_hist(x,y, values, x_bins, y_bins, statistic = 'count', 
                  figsize = (6,4), width_ratios = [1, 0.18, 0.05], height_ratios = [0.3,1], 
                  histogram_color = 'slategray', xlabel = '', ylabel = '', 
                  wspace = 0.05, hspace = 0.05,
                  grid_kwargs = {'alpha' : 0.3}, **kwargs):
    """
    Makes a 2d histogram with 1d histograms on sides. 
    x,y, values, x_bins, y_bins, statistic are passed directly to scipy.stats.binned_statistic_2d
    (note that values are not used when statistic = 'count', so that can be set to anything then)

    Third column in figure is for the colorbar.

    wspace, hspace determines the width and height between subplots.

    grid_kwargs are passed to plt.grid

    **kwargs are passed to plt.pcolormesh (for plotting 2d histogram)

    Returns (fig, axes)
    """

    # Make axes
    fig,axes = plt.subplots(2,3, width_ratios=width_ratios, height_ratios=height_ratios, figsize=figsize) 
    
    # 2D histogram:
    stats2d =  stats.binned_statistic_2d(x, y, values, statistic=statistic, bins=[x_bins, y_bins])
    # Replaces 0 in count with nan
    if statistic == 'count':
        stats2d.statistic[stats2d.statistic == 0] = np.nan
    im = axes[1,0].pcolormesh(stats2d.x_edge, stats2d.y_edge, stats2d.statistic.T, **kwargs)
    axes[1,0].set_xlabel(xlabel)
    axes[1,0].set_ylabel(ylabel)

    # 1D histogram x:
    axes[0,0].hist(x, bins=stats2d.x_edge, color = histogram_color);
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_xlim(stats2d.x_edge[[0,-1]])
    axes[0,0].xaxis.set_ticklabels([])

    # 1D histogram y:
    axes[1,1].hist(y, bins=stats2d.y_edge, orientation='horizontal', color = histogram_color);
    axes[1,1].set_xlabel('Count')
    axes[1,1].set_ylim(stats2d.y_edge[[0,-1]])
    axes[1,1].yaxis.set_ticklabels([])

    # Add colorbar
    fig.colorbar(im, cax=axes[1,2], label = statistic.capitalize())
    
    # Remove upper right axes
    axes[0,1].set_axis_off()
    axes[0,2].set_axis_off()

    # Gridlines
    axes[0,0].grid(**grid_kwargs)
    axes[1,0].grid(**grid_kwargs)
    axes[1,1].grid(**grid_kwargs)

    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    return fig, axes

def plot_twodstat(x,y,xbins=50,ybins=50,z=False,statistic="count",tickstep=False,axlines=(0,0),cmap = cmo.tempo, vmin=None, vmax=None, colorbar=True,meandot=True,meanline=False,axisequal=False, cbar_shrink = 1, norm=None, ax=None):
    """
    Compute and plot two a dimensional statistic. Copied from http://www.jmlilly.net/course/labs/html/VarianceEllipses-Python.html
    
    Args: 
        x: Array of x-values to be binned
        y: Array of y-values to be binned; same size as x
        
    Optional Args:
        xbins: Number of bins or array of bin edges for x-bins (default: 50)
        ybins: Number of bins or array of bin edges for y-bins (default: 50)
        z: Array of z-values for which statistic is be formed; same size as x 
        statistic: "count", "log10count", "mean", "median", or "std";
            defaults to "count", in which case the z argument is not needed
        tickstep: X- and y-axis tick step, a length 2 tuple; defaults to auto   
        axlines: Axis origin locations for horizontal and vertical lines,
            a length 2 tuple, defaults to (0,0), lines omitted if False
        cmap: Colormap, defaults to Spectral_r
        colorbar: Plots a colorbar, defaults to True
        meandot: Plots a dot at the mean value, defaults to true
        meanline: Plots a line from the origin to the mean value, defaults to false
        axisequal: Sets plot aspect ratio to equal, defaults to false

    Returns:
        im: Image handle    
        
    The computation of the statistic is handled by stats.binned_statistic_2d.
    
    Note for the computation of the standard deviation, we are using the form 
    <(z - <z>)^2> = <z^2> - <z>^2, which is much faster than the algorithm used 
    by stats.binned_statistic_2. 
    
    Note also that z may be complex valued, in which case we define the standard
    deviation the square root of <(z - <z>)(z - <z>)^*> = <|z|^2> - |<z>|^2,
    which will be real-valued and non-negative. 
    """
    valid_statistic = ["count", "log10count", "mean", "median", "std"]
    if statistic not in valid_statistic:
        raise ValueError(f"statistic must be one of {valid_statistic}")

    if norm == 'log':
        norm = colors.LogNorm()

    if isinstance(xbins, int):
        xbins = np.linspace(np.nanmin(x), np.nanmax(x), xbins+1)
        
    if isinstance(ybins, int):
        ybins = np.linspace(np.nanmin(y), np.nanmax(y), ybins+1)
    
    #plot just one twodhist    
    if statistic=="count":
        q = stats.binned_statistic_2d(x, y, None, bins=[xbins, ybins], statistic="count").statistic
        q[q==0]=np.nan  #swap zero values for NaNs, so they don't appear with a color
        clabel='Histogram'
    elif statistic=="log10count":
        q = stats.binned_statistic_2d(x, y, None, bins=[xbins, ybins], statistic="count").statistic
        q[q==0]=np.nan  #swap zero values for NaNs, so they don't appear with a color
        q=np.log10(q)
        clabel='Log10 Histogram'
    elif statistic=="mean":    
        q = stats.binned_statistic_2d(x, y, z, bins=[xbins, ybins], statistic="mean").statistic
        clabel='Mean'
    elif statistic=="median":    
        q = stats.binned_statistic_2d(x, y, z, bins=[xbins, ybins], statistic="median").statistic
        clabel='Median'
    elif statistic=="std":   
        #we are doing this ourselves because the algorithm used by binned_statistic_2d
        #is remarkably slow
        if np.all(np.isreal(z)): #real-valued case
            q2    = stats.binned_statistic_2d(x, y, z**2, bins=[xbins, ybins], statistic="mean").statistic
            qbar  = stats.binned_statistic_2d(x, y, z, bins=[xbins, ybins], statistic="mean").statistic
            q = np.sqrt(q2 - qbar**2)       
        else:  #complex-valued case
            q2    = stats.binned_statistic_2d(x, y, np.abs(z)**2, bins=[xbins, ybins], statistic="mean").statistic
            qbarr = stats.binned_statistic_2d(x, y, z.real, bins=[xbins, ybins], statistic="mean").statistic
            qbari = stats.binned_statistic_2d(x, y, z.imag, bins=[xbins, ybins], statistic="mean").statistic
            qbar = qbarr + 1j* qbari
            q = np.sqrt((q2 - np.abs(qbar)**2).real)        

        clabel='Standard Deviation'

    if ax is None:
        ax=plt.gca()
    
    im=ax.pcolormesh(xbins, ybins, np.transpose(q), cmap=cmap, shading="flat", vmin=vmin, vmax=vmax, norm=norm)
    if colorbar:
        cb=plt.gcf().colorbar(im, ax=ax, shrink=cbar_shrink)
        cb.set_label(clabel)
    
    if axisequal:
        ax.set_aspect("equal")
        
    if not(not(axlines)):
        ax.axhline(axlines[0], linestyle=":", color="k")
        ax.axvline(axlines[1], linestyle=":", color="k")

    if meanline:
        #plt.arrow(0,0,np.mean(x),np.mean(y),width=0.8,length_includes_head=False,facecolor="k",edgecolor="w")
        ax.plot([0,np.mean(x)],[0,np.mean(y)],color="w",linewidth=4.5)
        ax.plot([0,np.mean(x)],[0,np.mean(y)],color="k",linewidth=3)
            
    if meandot:
        ax.plot(np.mean(x),np.mean(y), "wo", markerfacecolor="k", markersize=8)

    plt.xlim([min(xbins), max(xbins)]),plt.ylim([min(ybins), max(ybins)])

    if not(tickstep==False):
        plt.xticks(np.arange(min(xbins), max(xbins), tickstep[0]))  # set x-label locations
        plt.yticks(np.arange(min(ybins), max(ybins), tickstep[1]))  # set x-label locations
    
    return im

def xr_plot_hist_with_mean_and_std(da, ax=None, **kwargs):
    """
    Plots histogram of DataArray with mean and standard deviation marked
    """
    if ax is None:
        ax = plt.gca()
    
    mean = da.mean()
    std = da.std()
    
    da.plot.hist(ax=ax, **kwargs)
    
    ylim = ax.get_ylim()
    ax.plot([mean, mean], ylim, 'k')
    ax.plot([mean-std, mean-std], ylim, 'k:')
    ax.plot([mean+std, mean+std], ylim, 'k:')
    ax.set_ylim(ylim)
        
    # place a text box in upper left in axes coords
    textstr = '\n'.join((r'$\mu=%.1e$' % (mean, ),
                         r'$\sigma=%.1e$' % (std, )))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,# fontsize=14,
            verticalalignment='top')    
    return

def mark_time_range(time_range, axis='x', ax = None, color = 'lavender', zorder=-2,  **kwargs):
    # Based on https://stackoverflow.com/a/31163913/11028793

    start = date2num(time_range[0])
    end = date2num(time_range[1])

    mark_range((start, end), axis=axis, ax=ax, color=color, zorder=zorder, **kwargs)

    return

def mark_range(range, axis='x', ax=None, color='lavender', zorder=-2, **kwargs):
    """
    Marks the given range in a plot with a rectangle.
    
    Example:
        mark_range((1,2), axis='y')
    """
    if ax is None:
        ax = plt.gca()

    if axis == 'x':      
        ylim = ax.get_ylim()
        width = range[1]-range[0]
        height = ylim[1]-ylim[0]
        lower_left_corner = (range[0], ylim[0])
    elif axis == 'y':
        xlim = ax.get_xlim()
        width = xlim[1]-xlim[0]
        height = range[1]-range[0]
        lower_left_corner = (xlim[0], range[0])
    else:
        raise ValueError("axis must be 'x' or 'y'")

    # Plot rectangle
    rect = Rectangle(lower_left_corner, width, height, zorder=zorder, color=color, **kwargs)
    ax.add_patch(rect)

    return

def mark_area(xlim, ylim, ax = None, scale=1, **kwargs):
    """
    Make a rectangle with extent xlim/ylim. The rectangle can be rescaled using the keyword 'scale' 
    (center point doesn't change).

    Example:
    mark_area([-1e5, 6e4], [2.1e6, 2.4e6], scale = 2, zorder=10, 
              transform=SouthPolarStereo(), 
              color = 'deeppink', fill=False, linewidth=2)
    """
    
    if ax is None:
        ax=plt.gca()

    # Rectangle dimensions
    width = (xlim[1] - xlim[0])*scale
    height = (ylim[1] - ylim[0])*scale
    
    # Anchor point
    center = (np.mean(xlim), np.mean(ylim))
    anchor = (center[0]-width/2,
              center[1]-height/2)

    ax.add_patch(Rectangle(anchor, width, height, **kwargs))
    

def make_colorbar(cmap, vmin, vmax, cax, **kwargs):
    """
    Make a colorbar in axis cax with specified cmap, vmin, vmax. 
    """
    norm = Normalize(vmin,vmax)
    plt.colorbar(ScalarMappable(norm, cmap=cmap),
                 cax=cax,
                 **kwargs)
    return


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Copied from https://stackoverflow.com/a/18926541
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
    
