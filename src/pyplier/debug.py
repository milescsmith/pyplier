import numpy as np
import pandas as pd
from rich.console import Console

from pyplier.num_pc import num_pc

console = Console()
# rng = np.random.default_rng()
# mat = rng.standard_normal([1000,1000])

# np.savetxt("test_pc_mat_1000.txt", mat, delimiter=",")
mat = np.loadtxt(
    "/Users/milessmith/workspace/pyplier/tests/data/num_pc/test_pc_mat_100.csv",
    delimiter=",",
)
# mat = np.loadtxt("/Users/milessmith/workspace/pyplier/tests/data/num_pc/test_pc_mat_1000.csv", delimiter=",")
# mat = pd.read_csv("/Users/milessmith/workspace/variance_stabilized_expression.csv.gz", index_col=0)
console.print(f"[red bold]{num_pc(mat, method='elbow')}[/red bold]", style="blink")
