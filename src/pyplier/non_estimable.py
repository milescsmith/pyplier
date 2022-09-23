import pandas as pd
from scipy.linalg import qr


def nonEstimable(x: pd.DataFrame) -> list[str] | None:
    p = x.shape[1]

    q, _, pivot = qr(x, pivoting=True)
    qrank = q.shape[0]

    if qrank >= p:
        return None
    n = x.columns.values

    if n is None:
        n = [str(_) for _ in range(p)]

    notest = n[pivot[(qrank + 1) : p]]

    return [str(x) if y == "" else y for x, y in enumerate(notest)] if any(_ == "" for _ in notest) else notest
