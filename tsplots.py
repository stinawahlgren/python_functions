import numpy as np
import matplotlib.pyplot as plt
from gsw import sigma0, CT_freezing

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
    
def freezing_line(SA_lim, p, N_SA = 30, saturation_fraction = 0, linestyle = 'dashed', color = 'k', **kwargs):
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
    SA = np.linspace(SA_lim[0], SA_lim[1], N_SA)
    freezing_line = CT_freezing(SA, p, saturation_fraction)
    plt.plot(SA, freezing_line, linestyle = linestyle, color = color,**kwargs)
