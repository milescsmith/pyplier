from typing import Dict, List

from pandas import DataFrame

from dataclasses import dataclass

@dataclass
class PLIERResults:
    residual: DataFrame = None
    B: DataFrame = None
    Z: DataFrame = None
    U: DataFrame = None
    C: DataFrame = None
    L1: float = None
    L2: float = None
    L3: float = None
    heldOutGenes: Dict[str, List[str]] = None
    withPrior: Dict[str, int] = None
    Uauc: DataFrame = None
    Up: DataFrame = None
    summary: DataFrame = None
