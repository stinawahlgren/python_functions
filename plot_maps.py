from cartopy.feature import NaturalEarthFeature, LAND
from cartopy.crs import PlateCarree, SouthPolarStereo
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np


def plot_antarctica(land_color='slategray', iceshelf_color='lightsteelblue', max_latitude=-60, make_round=True):
    """
    Uses cartopy and Natural Earth to make a figure over Antarctica including ice shelves.
    """

    # Make plot
    ax = plt.axes(projection = SouthPolarStereo())
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

    return ax
