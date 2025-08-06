from pathlib import Path

import pandas as pd
import numpy as np

# from pyplier import PLIER
from pyplier.solve_u import solve_u

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

data_dir = Path.home().joinpath("workspace", "pyplier", "tests", "data")
solve_u_dir = data_dir.joinpath("solve_u")
test_z = pd.read_csv(solve_u_dir.joinpath("z.csv.gz"), index_col="gene")
test_chat = pd.read_csv(solve_u_dir.joinpath("chat.csv.gz"), index_col="pathway")
test_prior_mat = prior_mat = pd.read_csv(solve_u_dir.joinpath("prior_mat.csv.gz"), index_col="gene")
prior_mat.columns.name = "pathway"
test_penalty_factor = np.loadtxt(solve_u_dir.joinpath("penalty_factor.csv.gz"))
expected_u_complete = pd.read_csv(solve_u_dir.joinpath("u_complete.csv.gz"), index_col="pathway").astype(np.float64)
expected_u_fast = pd.read_csv(solve_u_dir.joinpath("u_fast.csv.gz"), index_col="pathway").astype(np.float64)


u_result = solve_u(
    z=test_z,
    chat=test_chat,
    prior_mat=test_prior_mat,
    penalty_factor=test_penalty_factor,
    pathway_selection="complete",
    glm_alpha=0.9,
    max_path=10,
    target_frac=0.7,
    l3=None,
)

# pbmc3k_x = pd.read_csv(common_data_dir / "pbmc3k_x.csv.gz", index_col=0)
# reactome = pd.read_csv(common_data_dir / "reactome.csv.gz", index_col="genes")

# testres = PLIER(pbmc3k_x.T, reactome, seed=4111950)
