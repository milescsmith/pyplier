from pathlib import Path

import pandas as pd

from pyplier import combinePaths, PLIER

common_data_dir = Path("/home/milo/workspace/pyplier")

# bloodCellMarkersIRISDMAP = pd.read_csv(
#     common_data_dir / "bloodCellMarkersIRISDMAP.csv.gz", index_col="gene"
# )
# canonicalPathways = pd.read_csv(
#     common_data_dir / "canonicalPathways.csv.gz", index_col="gene"
# )
# dataWholeBlood = pd.read_csv(
#     common_data_dir / "dataWholeBlood.csv.gz", index_col="gene"
# )

pbmc3k_x = pd.read_csv(common_data_dir / "pbmc3k_x.csv.gz", index_col=0)
reactome = pd.read_csv(common_data_dir / "reactome.csv.gz", index_col="genes")

testres = PLIER(pbmc3k_x.T, reactome, seed=4111950)
