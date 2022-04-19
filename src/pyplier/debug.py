from pathlib import Path

import pandas as pd

from pyplier.combinePaths import combinePaths
from pyplier.plier import PLIER

common_data_dir = Path("/workspaces/pyplier", "tests", "data", "common")

bloodCellMarkersIRISDMAP = pd.read_csv(
    common_data_dir / "bloodCellMarkersIRISDMAP.csv.gz", index_col="gene"
)
canonicalPathways = pd.read_csv(
    common_data_dir / "canonicalPathways.csv.gz", index_col="gene"
)
dataWholeBlood = pd.read_csv(
    common_data_dir / "dataWholeBlood.csv.gz", index_col="gene"
)

allPaths = combinePaths(bloodCellMarkersIRISDMAP, canonicalPathways)
testres = PLIER(dataWholeBlood, allPaths, seed=4111950)
