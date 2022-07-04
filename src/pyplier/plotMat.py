# from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import seaborn as sns

# from .utils import colNorm
from typeguard import typechecked


@typechecked
def plotMat(
    mat: pd.DataFrame,
    scale: bool = True,
    trim_names: Optional[int] = 50,
    cutoff: Optional[float] = None,
    *args,
    **kwargs
) -> Dict[str, np.ndarray]:
    if trim_names is not None:
        mat.index = mat.index.str.slice(stop=trim_names)
        mat.columns = mat.columns.str.slice(stop=trim_names)

    if cutoff is not None:
        mat.where(cond=lambda x: x > cutoff, other=0, inplace=True)

    iirow = np.where(mat.abs().sum(axis=1) > 0)[0]
    mat = mat.iloc[iirow, :]
    iicol = np.where(mat.abs.sum(axis=0) > 0)[0]
    mat = mat.iloc[:, iicol]

    if scale:
        aa = mat.abs().max(axis=0)  # colMax
        aa.replace(cond=lambda x: x != 0, other=0, inplace=True)
        mat = mat.divide(other=aa, axis=0)

    sns.clustermap(data=mat, *args, **kwargs)

    return {"iirow": iirow, "iicol": iicol}
