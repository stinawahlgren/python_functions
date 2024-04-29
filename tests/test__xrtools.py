from ..xrtools import has_same_coords, da_from_binned_statistic_1d

import xarray as xr
import numpy as np


def test__has_same_coords():
    
    da1 = xr.DataArray([[25, 35], [10, 24]],
                       dims=("x", "y"),
                       coords={"x": [35.0, 40.0], "y": [100.0, 120.0]})
    da2 = xr.DataArray([[25, 35], [10, 24]],
                       dims=("y", "x"),
                       coords={"x": [35.0, 40.0], "y": [100.0, 120.0]})    
    da3 = xr.DataArray([25, 35],
                       dims=("y"),
                       coords={"y": [100.0, 120.0]})
    da4 = xr.DataArray([[25, 35], [10, 24]],
                       dims=("y", "x"),
                       coords={"x": [0, 40.0], "y": [100.0, 120.0]})
    da5 = xr.DataArray([[25, 35, 2], [10, 24, 10]],
                       dims=("x", "y"),
                       coords={"x": [35.0, 40.0], "y": [100.0, 120.0, 123.0]})    
    
    # Same coordinates
    assert has_same_coords(da1,da1), "Those have identical coordinates, should be True"
    
    # Different order of dimensions
    assert has_same_coords(da1,da2), "Should be True even if the order of the dimensions are different"
    
    # Common coordinates match, but one has an extra dimension
    assert not has_same_coords(da1,da3), "da1 have one more dimension than da3, they should not be considered equal"
    
    # Same dimensions, but coordinate values differ
    assert not has_same_coords(da1,da4), "The coordinates have different values, they should not be considered equal"
    
    # Different coordinate length
    assert not has_same_coords(da1,da5), "The length of the coordinates differ, they should not be considered equal"



def test__da_from_binned_statistics_1d():
    
    x = xr.DataArray( np.linspace(-1,1,10), dims='time')
    y = xr.DataArray(
        np.array([0.08026983, 0.12812636, 0.12214241, 0.22667363, 0.14381985,
                  0.35799247, 0.03849258, 0.63907954, 0.47213111, 0.89872314]),
        dims = 'time')
    x_grid = xr.DataArray([-0.9,0,0.2,0.5],
                          dims = 'z',
                          )

    da_count_expected = xr.DataArray([3,2,2,1], dims='z', coords = {'z': [-0.9,0,0.2,0.5]})
    assert da_count_expected.equals(da_from_binned_statistic_1d(x, y, x_grid, 'count'))
