import numpy as np
import matplotlib.pyplot as plt
from gsw import sigma0, CT_freezing, melting_ice_SA_CT_ratio

def density_contour_lines(SA_lim, CT_lim, N = 10, N_SA = 100, N_CT = 100, colors = 'k', linewidths=0.5, **kwargs):
    """
    TS-plot contour lines with density anomalies. (Conservative temperature vs absolute salinity) Density anomalies are computed using the function sigma0 from Gibbs See Water package. 
    
    Parameters:
        SA_lim : Min and max absolute salinity
        CT_lim : Min and max conservative temperature
        N      : Number of contour lines (default 10)
        N_SA   : Number of points in salinity grid (default 100)
        N_CT   : Number of points in temperature grid
        
        + keyword arguments passed to matplotlib.pyplot.contour
        
    Example usage: 
    - If TS-plot already plotted: 
        density_contour_lines(plt.xlims(), plt.ylims())
    - With explicit limits:
        density_contour_lines([34, 35], [-2, 5])
    """  
    # Create meshgrid
    (SA, CT) = np.meshgrid(np.linspace(SA_lim[0], SA_lim[1], 101),
                           np.linspace(CT_lim[0], CT_lim[1], 100),
                           indexing='xy')
    
    # Compute density anomaly
    sigma = sigma0(SA,CT)

    # Plot  
    im = plt.contour(SA,CT,sigma, N, colors = colors, linewidths=linewidths, **kwargs)
    plt.gca().clabel(im, fontsize=10, inline=True)
    plt.ylabel('Conservative temperature (°C)')
    plt.xlabel('Absolute salinity (g/kg)')
    
def freezing_line(SA_lim, p, N_SA = 30, saturation_fraction = 0,  fill_below = False, **plot_kwargs):
    """
    Mark freezing line in TS-plot (conservative temperature vs absolute salinity). The freezing point is calculated using the function CT_freezing from Gibbs See Water package.
    
    Parameters:
        SA_lim : Min and max absolute salinity
        p      : Sea pressure
        N_SA   : Number of points in salinity grid (default 30)
        saturation_fraction : The saturation fraction of dissolved air in seawater (default 0)
        
        + keyword arguments passed to matplotlib.pyplot.plot
        
    Example usage: 
    - If TS-plot already plotted: 
        freezing_line(plt.xlims(), 0)
    - With explicit limits:
        freezing_line([0, 35], 0)
    """
    ylim = plt.ylim()
    SA = np.linspace(SA_lim[0], SA_lim[1], N_SA)
    freezing_line = CT_freezing(SA, p, saturation_fraction)

    if fill_below:
        plt.gca().fill_between(SA, freezing_line, ylim[0]-1,  **plot_kwargs)
    else:
        kwargs  = {'linestyle' : 'dashed', 'color' : 'k'}
        kwargs.update(plot_kwargs)
        plt.plot(SA, freezing_line, **kwargs)
    plt.ylim(ylim)

def plot_gade_line(SA_amb, CT_amb, p, t_ice, SA_min=None, **kwargs):

    if SA_min is None:
        SA_min = gade_freezing_line_intersect(SA_amb, CT_amb, p, t_ice)[0]
       
    dCT_dSA = 1/melting_ice_SA_CT_ratio(SA_amb, CT_amb, p, t_ice)
    SA_gade = [SA_amb, SA_min]
    CT_gade = [CT_amb, CT_amb + dCT_dSA*(SA_min-SA_amb)]
    plt.plot(SA_gade, CT_gade, **kwargs)


def gade_freezing_line_intersect(SA_amb, CT_amb, p, t_ice, saturation_fraction=0, SA_diff=-5):
    """
    Find where the Gade line crosses the freezing line. Freezing line is linearized
    based on SA_amb and SA_amb + SA_diff
    """

    slope_gade = 1/melting_ice_SA_CT_ratio(SA_amb, CT_amb, p, t_ice)
    intercept_gade = CT_amb - slope_gade * SA_amb
    
    freezing_line = CT_freezing([SA_amb, SA_amb + SA_diff], p, saturation_fraction)
    slope_freezing = (freezing_line[1]-freezing_line[0])/SA_diff
    intercept_freezing = freezing_line[0] - slope_freezing * SA_amb
    
    SA_intersect = (intercept_freezing - intercept_gade) / (slope_gade - slope_freezing)
    CT_intersect = intercept_gade + slope_gade*SA_intersect

    return(SA_intersect, CT_intersect)
