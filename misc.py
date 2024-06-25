import numpy as np

def get_edges(centers):
    centers = np.array(centers)
    mid = centers[:-1] + (centers[1:]-centers[:-1])/2
    first = centers[0] - (centers[1]-centers[0])/2
    last  = centers[-1] + (centers[-1]-centers[-2])/2
    return np.concatenate([[first], mid, [last]])

def get_centers(edges):
    edges = np.array(edges)
    centers = (edges[:-1]+edges[1:])/2
    return centers

def all_columns_equal(a):
    """
    Check if all columns in a are equal
    
    Parameters:
        a: (n,m) numpy.ndarray
        
    Returns:
        True/False
    """
    m = a.shape[1]
    return ~(a - np.tile(a[:,0], (m,1)).T).any()
