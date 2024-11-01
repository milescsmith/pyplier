from typing import TypeVar

import numpy as np
from rich import print as rprint

from pyplier.plier_res import PLIERResults

PLIERRes = TypeVar("PLIERRes", bound="PLIERResults")


def name_b(plierRes: PLIERResults, top: int = 1, fdr_cutoff: float = 0.01, use: str | None = None) -> list[str]:
    """
    Rename latent variables to match the pathways that appear to correlate
    Number of pathways used in the name is controlled by `top`
    """
    if use is None:
        use = "coef"
    elif use not in ("coef", "AUC"):
        msg = "only 'coef' and 'AUC' are the only valid options for the 'use' argument"
        raise ValueError(msg)

    names = []

    if use == "coef":
        uuse = plierRes.u.copy(deep=True)
    else:
        uuse = plierRes.uauc.copy(deep=True)

    if plierRes.up is not None:
        pval_cutoff = plierRes.summary.loc[plierRes.summary["FDR"] < fdr_cutoff, "p-value"].max()
        if np.isnan(pval_cutoff):
            rprint("[red]No p-values in PLIER object were below the fdr_cutoff: using coefficients only[/]")
        else:
            uuse[plierRes.up > pval_cutoff] = 0

    else:
        rprint("[red]No p-values in PLIER object: using coefficients only[/]")

    mm = uuse.apply(func=np.max, axis=0)

    for i in range(plierRes.u.shape[1]):
        if mm.iloc[i] > 0:
            names.append(
                f"{i + 1}," + ",".join(uuse.iloc[:, i].sort_values(ascending=False).where(lambda x: x > 0).index[:top])
            )
        # this should give us something like "LV1,REACTOME_GENERIC_TRANSCRIPTION_PATHWAY"
        # this also will only return pathways with some correlation - if there is 0, it will get dropped and the
        # [0:top] is ignored, grabbing just as much as it can
        elif max(plierRes.u.iloc[:, i]) > 0:
            names.append(
                f"{i + 1},"
                + ",".join(plierRes.u.iloc[:, i].sort_values(ascending=False).where(lambda x: x > 0).index[:top])
            )
        else:
            names.append(f"LV {i+1}")

    return names
