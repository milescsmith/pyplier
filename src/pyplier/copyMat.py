import numpy as np
import pandas as pd

def copyMat(df: pd.DataFrame, zero:bool = False) -> pd.DataFrame:
    if zero:
        dfnew = pd.DataFrame(np.zeros(shape=df.shape), index=df.index, columns=df.columns)
    else:
        dfnew = df.copy(deep=True)
    
    return dfnew