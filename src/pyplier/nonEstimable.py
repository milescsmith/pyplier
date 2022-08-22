from typing import Optional, List

import pandas as pd
from scipy.linalg import qr


def nonEstimable(x: pd.DataFrame) -> Optional[List[str]]:

    p = x.shape[1]

    q, _, pivot = qr(x, pivoting=True)
    qrank = q.shape[0]

    if qrank < p:
        n = x.columns.values

        if n is None:
            n = [str(_) for _ in range(p)]

        notest = n[pivot[(qrank + 1) : p]]

        if any([_ == "" for _ in notest]):
            return [str(x) if y == "" else y for x, y in enumerate(notest)]
        else:
            return notest
    else:
        return None
