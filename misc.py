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

def dataset_metadata2latex_table(ds, type='variables'):
    """
    type: 'variables'/'coords'/'both'

    Outputs some of the metadata of an xarray.Dataset to latex tabular format.
    Note that the latex package makecell is required if description contains 
    line breaks. (Add \\usepackage{makecell} to the tex-document)
    """

    if type == 'variables':
        keys = list(ds.data_vars)
    elif type == 'coords':
        keys = list(ds.coords)
    elif type == 'both':
        keys = list(ds.coords) + list(ds.data_vars)
    
    print('\\begin{tabular}{llll}')
    print('\\textbf{Variable} & \\textbf{Description} & \\textbf{Unit} & \\textbf{Dimension} \\\\')
    print('\\hline')
    for var in keys:
        if 'unit' in ds[var].attrs.keys():
            unit = ds[var].attrs['unit']
        else:
            unit = ''

        # fix line breaks in description
        description = ds[var].attrs["description"]
        if '\n' in description:
            description = '\\makecell[l]{' + description.replace('\n', '\\\\') + '}'
        
        # Remove '' from dimension
        dims = str(ds[var].dims).replace("'","")

        # Remove traling , from dimension
        if dims[-2] == ',':
            dims = dims[:-2]+')'

        # Replace _ with \_
        row = (f'{var} & {description} & {unit} & {dims}\\\\').replace('_', '\\_')
        print(row)
    print('\\end{tabular}')
    return
