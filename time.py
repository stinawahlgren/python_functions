import pandas as pd

def matlabtime2datetime(t):
    """
    Converts from Matlab time to datetime using pandas.to_datetime.

    example:

    >>> matlabtime2datetime(739259.147573)

    Timestamp('2024-01-08 03:32:30.307204570')   
    """
    return pd.to_datetime(t-719529,unit='D')
