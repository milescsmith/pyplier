from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm


def AUC(labels: pd.Series, values: pd.Series) -> Dict[str, float]:
    posii = labels[labels > 0]
    negii = labels[labels <= 0]
    posn = len(posii)
    negn = len(negii)
    posval = values[posii.index]
    negval = values[negii.index]
    if posn > 0 and negn > 0:
        statistic, pvalue = mannwhitneyu(posval, negval, alternative="greater")
        conf_int_low, conf_int_high = mannwhitneyu_conf_int(posval, negval)
        res = {
            "low": conf_int_low,
            "high": conf_int_high,
            "auc": (statistic / (posn * negn)),
            "pval": pvalue,
        }
    else:
        res = {"auc": 0.5, "pval": np.nan}

    return res


def mannwhitneyu_conf_int(
    x: np.array, y: np.array, alpha: float = 0.05
) -> Tuple[float, float]:
    lx = len(x)
    ly = len(y)

    N = norm.ppf(1 - alpha / 2)

    diffs = sorted([i - j for i in x for j in y])

    # For an approximate 100(1-a)% confidence interval first calculate K:
    k = int(round(lx * ly / 2 - (N * (lx * ly * (lx + ly + 1) / 12) ** 0.5)))

    # The Kth smallest to the Kth largest of the n x m differences
    # lx and ly should be > ~20
    CI = (diffs[k], diffs[len(diffs) - k])
    return CI
