from typing import Dict

import pandas as pd
import numpy as np


def getCutoff(aucRes: Dict[str, pd.DataFrame], fdr_cutoff: float = 0.01) -> float:
    return np.amax(aucRes["summary"][aucRes["summary"]["FDR"] <= fdr_cutoff]["p-value"])
