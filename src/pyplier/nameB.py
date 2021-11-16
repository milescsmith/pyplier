import numpy as np

from .console import console
from .stubs import PLIERResults


def nameB(
    plierRes: PLIERResults, top: int = 1, fdr_cutoff: float = 0.01, use: str = None
) -> list[str]:
    """
    Rename latent variables to match the pathways that appear to correlate
    Number of pathways used in the name is controlled by `top`
    """
    if use is None:
        use = "coef"
    elif use not in ("coef", "AUC"):
        raise ValueError(
            "only 'coef' and 'AUC' are the only valid options for the 'use' argument"
        )

    names = list()

    if use == "coef":
        Uuse = plierRes["U"].copy(deep=True)
    else:
        Uuse = plierRes["Uauc"].copy(deep=True)

    if plierRes["Up"] is not None:
        pval_cutoff = max(
            plierRes["summary"].loc[plierRes["summary"]["FDR"] < fdr_cutoff, "p-value"]
        )
        Uuse[plierRes["Up"] > pval_cutoff] = 0
    else:
        console("[red]No p-values in PLIER object: using coefficients only[/]")

    mm = Uuse.apply(func=np.max, axis=0)

    for i in range(plierRes["U"].shape[1]):
        if mm[i] > 0:
            names.append(
                f"{i+1},"
                + ",".join(
                    Uuse.iloc[:, i]
                    .sort_values(ascending=False)
                    .where(lambda x: x > 0)
                    .index[0:top]
                )
            )
            # this should give us something like "LV1,REACTOME_GENERIC_TRANSCRIPTION_PATHWAY"
            # this also will only return pathways with some correlation - if there is 0, it will get dropped and the
            # [0:top] is ignored, grabbing just as much as it can
        elif max(plierRes["U"].iloc[:, i]) > 0:
            names.append(
                f"{i+1},"
                + ",".join(
                    plierRes["U"]
                    .iloc[:, i]
                    .sort_values(ascending=False)
                    .where(lambda x: x > 0)
                    .index[0:top]
                )
            )
        else:
            names.append(f"LV {i+1}")

    return names
