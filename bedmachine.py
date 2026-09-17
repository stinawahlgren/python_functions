import numpy as np
from xarray import load_dataset
from cartopy.crs import PlateCarree, SouthPolarStereo, Globe

def read_bedmachine(path = '../../data/bedmachine/NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc'):
    return load_dataset(path)

def plot_bedmachine(ds, var, limits, step=10, contour=False, mask = None, **kwargs):
    """
    Parameters:
        ds     : bedmachine data
        var    : variable from bedmahcine to plot: 
                    'mask'/'firn'/'surface'/'thickness'/'bed'/'errbed'/'source'/'dataid'/'geoid'/'rgi'
        limits : [lon_min, lon_max, lat_min, lat_max]
        step   : plot every 'step' point
        **kwargs passed to xarray.DataArray.plot()
    """
    # Bedmachine projection
    proj_bedmachine = SouthPolarStereo(central_longitude = 0, 
                                       true_scale_latitude = -71,
                                       globe = Globe(ellipse='WGS84'))
    
    # Transform corners to bedmahcine polar stereographic projection:
    N = 50
    lon_line = np.linspace(limits[0], limits[1], N)
    lons = np.concatenate([lon_line, [limits[1]]*N, lon_line, [limits[0]]*N])
    lat_line = np.linspace(limits[2], limits[3], N)
    lats = np.concatenate([[limits[2]]*N, lat_line, [limits[3]]*N, lat_line])
    boundary_xy = proj_bedmachine.transform_points(PlateCarree(), lons, lats)

    margin = 2e3   
    xmin = np.min(boundary_xy[:,0]) - margin
    xmax = np.max(boundary_xy[:,0]) + margin
    ymin = np.min(boundary_xy[:,1]) - margin
    ymax = np.max(boundary_xy[:,1]) + margin

    da = ds[var].sel(x=slice(xmin,xmax), y=slice(ymax,ymin)
                    ).isel(x=slice(None,None,step), y=slice(None,None,step))

    # Mask
    if mask is not None:
        da = da.where(
            mask.sel(
                x=slice(xmin,xmax), y=slice(ymax,ymin)
                     ).isel(
                         x=slice(None,None,step), y=slice(None,None,step)
                     )
            )
                                
    # Plot
    if contour:
        im = da.plot.contour(transform=proj_bedmachine, **kwargs)
    else:
        im = da.plot(transform=proj_bedmachine, **kwargs)
    return im
