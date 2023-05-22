from genericpath import isfile
import os 

import pandas as pd
#def dataframe_from_csv(path, header=0, index_col=0):
#    return pd.read_csv(path, header=header, index_col=index_col)

def dataframe_from_csv(path, header=0, index_col=0):
    if os.path.isfile(path):
        return pd.read_csv(path, header=header, index_col=index_col)    
    return pd.read_csv(path+".gz", compression='gzip', header=header, index_col=index_col)