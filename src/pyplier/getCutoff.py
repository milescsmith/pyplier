from typing import Dict
import pandas as pd

from .stubs import PLIERResults


def getCutoff(aucRes: Dict[str, pd.DataFrame], fdr_cutoff: float = 0.01) -> float:
    return max(aucRes["summary"][aucRes["summary"]["FDR"] <= fdr_cutoff]["p-value"])
