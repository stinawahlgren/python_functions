from cartopy.feature import NaturalEarthFeature, LAND
from cartopy.crs import PlateCarree, SouthPolarStereo, TransverseMercator 
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker

import numpy as np


def plot_antarctica(land_color='slategray', iceshelf_color='lightsteelblue', max_latitude=-60, make_round=True, central_longitude=0, transparent_border = True, hide_frame=False):
    """
    Uses cartopy and Natural Earth to make a figure over Antarctica including ice shelves.
    """

    # Make plot
    ax = plt.axes(projection = SouthPolarStereo(central_longitude=central_longitude))
    ax.set_extent([-180, 180, -90, max_latitude], PlateCarree())
    
    # Plot ice shelves
    iceshelves = NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '10m',
                                     edgecolor = 'face',
                                     facecolor = iceshelf_color)
    ax.add_feature(iceshelves, zorder=-1);
    
    # Plot land
    ax.add_feature(LAND, zorder=0, edgecolor='face', facecolor=land_color)
    
    if make_round:
        # Based on this example: https://scitools.org.uk/cartopy/docs/v0.15/examples/always_circular_stereo.html
        # Compute a circle in axes coordinates, which we can use as a boundary
        # for the map. We can pan/zoom as much as we like - the boundary will be
        # permanently circular.
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

    if transparent_border:
        # Make outside of plot transparent
        plt.gcf().set_facecolor([0,0,0,0])

    if hide_frame:
        ax.spines[:].set_visible(False)
        
    return ax


def nice_lonlat_gridlines(ax, longitudes=None, latitudes=None, size=8, linewidth=0.5, color='lightgrey', labels = ['bottom', 'left'], **kwargs):
    """
    Makes longitude/latitude ticks nice and small. Cartopys polar stereographic 
    ticks are otherwise annoyingly large and weirdly rotated.

    Based on https://stackoverflow.com/a/65382042
    """

    # Make gridlines
    gl = ax.gridlines(draw_labels=True,x_inline=False,y_inline=False, crs=PlateCarree(), linewidth=linewidth, color=color, **kwargs)

    # Control tick gridline location
    if longitudes is not None:
        gl.xlocator = mticker.FixedLocator(longitudes)
    if latitudes is not None:
        gl.ylocator = mticker.FixedLocator(latitudes)

    # Only show ticks on specified axes
    gl.bottom_labels = 'bottom' in labels
    gl.left_labels   = 'left' in labels
    gl.top_labels    = 'top' in labels
    gl.right_labels  = 'right' in labels

    # Adjust rotation and size of tick labels
    gl.xlabel_style['size']=size
    gl.xlabel_style['rotation']=0
    gl.xlabel_style['ha'] = 'center'
    
    gl.ylabel_style['size']=size
    gl.ylabel_style['rotation']=90
    gl.ylabel_style['ha'] = 'center'

    return gl


def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3, fontsize=None, textoffset=0, zorder=10):
    """
    Copied and slightly modified from: https://stackoverflow.com/a/35705477
    
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.

    textoffset adds an extra vertical distance between text and bar
    """
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length: 
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)        
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length) 

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx, sbx + length * 1000]

    # Save xlim, ylim to later:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth, zorder=zorder)
    #Plot the scalebar label
    text = ax.text(sbx + length * 500, sby+textoffset, str(length) + ' km', transform=tmc,
                   horizontalalignment='center', verticalalignment='bottom', zorder=zorder)

    # Restore original axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Set font size
    if fontsize is not None:
        text.set_size(fontsize)
