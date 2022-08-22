import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from rich import print as rprint
from tqdm.auto import tqdm


# TODO: should we keep the original source data as part of this object?
# TODO: should we keep the priorMat as part of this object?  Maybe make `PLIER` a member function?
class PLIERResults(object):
    def __init__(
        # TODO: consider making some of these csc_matrices
        self,
        B: pd.DataFrame = pd.DataFrame(),
        Z: pd.DataFrame = pd.DataFrame(),
        U: pd.DataFrame = pd.DataFrame(),
        C: pd.DataFrame = pd.DataFrame(),
        L1: float = 0.0,
        L2: float = 0.0,
        L3: float = 0.0,
        heldOutGenes: Dict[str, List[str]] = defaultdict(list),
        withPrior: Dict[str, int] = defaultdict(int),
        Uauc: pd.DataFrame = pd.DataFrame(),
        Up: pd.DataFrame = pd.DataFrame(),
        summary: pd.DataFrame = pd.DataFrame(),
        residual: pd.DataFrame = pd.DataFrame(),
    ):
        self.residual = residual
        self.B = B
        self.Z = Z
        self.U = U
        self.C = C
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.heldOutGenes = heldOutGenes
        self.withPrior = withPrior
        self.Uauc = Uauc
        self.Up = Up
        self.summary = summary

    def __repr__(self) -> str:
        return (
            f"B : {self.B.shape[0]} rows x {self.B.shape[1]} columns\n"
            f"Z : {self.Z.shape[0]} rows x {self.Z.shape[1]} columns\n"
            f"U : {self.U.shape[0]} rows x {self.U.shape[1]} columns\n"
            f"C : {self.C.shape[0]} rows x {self.C.shape[1]} columns\n"
            f"heldOutGenes: {len(self.heldOutGenes)}\n"
            f"withPrior: {len(self.withPrior)}\n"
            f"Uauc: {self.Uauc.shape[0]} rows x {self.Uauc.shape[1]} columns\n"
            f"Up: {self.Up.shape[0]} rows x {self.Up.shape[1]} columns\n"
            f"summary: {self.summary.shape[0]} rows x {self.summary.shape[1]} columns\n"
            f"residual: {self.residual.shape[0]} rows x {self.residual.shape[1]} columns\n"
            f"L1 is set to {self.L1:.4f}\n"
            f"L2 is set to {self.L2:.4f}\n"
            f"L3 is set to {self.L3:.4f}"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other) -> bool:
        if (
            np.isclose(self.B, other.B).all()
            and np.isclose(self.Z, other.Z).all()
            and np.isclose(self.U, other.U).all()
            and np.isclose(self.C, other.C).all()
            and self.L1 == other.L1
            and self.L2 == other.L2
            and self.L3 == other.L3
            and self.heldOutGenes == other.heldOutGenes
            and self.withPrior == other.withPrior
            and np.isclose(self.Uauc, other.Uauc).all()
            and np.isclose(self.Up, other.Up).all()
            and np.isclose(self.summary, other.summary).all()
            and np.isclose(self.residual, other.residual).all()
        ):
            return True
        else:
            if not np.isclose(self.B, other.B).all():
                rprint("[red]B[/red] is unequal")
            if not np.isclose(self.Z, other.Z).all():
                rprint("[red]Z[/red] is unequal")
            if not np.isclose(self.U, other.U).all():
                rprint("[red]U[/red] is unequal")
            if not np.isclose(self.C, other.C).all():
                rprint("[red]C[/red] is unequal")
            if self.L1 != other.L1:
                rprint("[red]L1[/red] is unequal")
            if self.L2 != other.L2:
                rprint("[red]L2[/red] is unequal")
            if self.L3 != other.L3:
                rprint("[red]L3[/red] is unequal")
            if self.heldOutGenes != other.heldOutGenes:
                rprint("[red]heldOutGenes[/red] is unequal")
            if self.withPrior != other.withPrior:
                rprint("[red]withPrior[/red] is unequal")
            if not np.isclose(self.Uauc, other.Uauc).all():
                rprint("[red]Uauc[/red] is unequal")
            if not np.isclose(self.Up, other.Up).all():
                rprint("[red]Up[/red] is unequal")
            if not np.isclose(self.summary, other.summary).all():
                rprint("[red]summary[/red] is unequal")
            if not np.isclose(self.residual, other.residual).all():
                rprint("[red]residual[/red] is unequal")
            return False

    def to_dict(self):
        return {
            "B": self.B.to_dict(),
            "Z": self.Z.to_dict(),
            "U": self.U.to_dict(),
            "C": self.C.to_dict(),
            "L1": self.L1,
            "L2": self.L2,
            "L3": self.L3,
            "heldOutGenes": self.heldOutGenes,
            "withPrior": self.withPrior,
            "residual": self.residual.to_dict(),
            "Uauc": self.Uauc.to_dict(),
            "Up": self.Up.to_dict(),
            "summary": self.summary.to_dict(),
        }

    def to_disk(self, loc: Path, compress: bool = False) -> bool:
        if not isinstance(loc, Path):
            loc = Path(loc)
        if compress or loc.suffix == ".gz":
            with gzip.open(loc, "wt", encoding="UTF-8") as zipfile:
                json.dump(self.to_dict(), zipfile)
        else:
            with open(loc, "w") as jsonfile:
                json.dump(obj=self.to_dict(), fp=jsonfile)
        return True

    @classmethod
    def from_dict(cls, source):
        pr = cls()
        if "B" in source:
            pr.B = pd.DataFrame.from_dict(source["B"])
        if "Z" in source:
            pr.Z = pd.DataFrame.from_dict(source["Z"])
        if "U" in source:
            pr.U = pd.DataFrame.from_dict(source["U"])
        if "C" in source:
            pr.C = pd.DataFrame.from_dict(source["C"])

        if "L1" in source:
            pr.L1 = source["L1"]
        if "L2" in source:
            pr.L2 = source["L2"]
        if "L3" in source:
            pr.L3 = source["L3"]

        if "heldOutGenes" in source:
            pr.heldOutGenes = source["heldOutGenes"]
        if "withPrior" in source:
            pr.withPrior = source["withPrior"]

        if "residual" in source:
            pr.residual = pd.DataFrame.from_dict(source["residual"])
        if "Uauc" in source:
            pr.Uauc = pd.DataFrame.from_dict(source["Uauc"])
        if "Up" in source:
            pr.Up = pd.DataFrame.from_dict(source["Up"])
        if "summary" in source:
            pr.summary = pd.DataFrame.from_dict(source["summary"])
        return pr

    @classmethod
    def from_disk(cls, loc: Path):
        """TODO: switch to using something like HDF5 or parquet for storage"""
        if str(loc).endswith(".gz"):
            with gzip.open(loc, "rt", encoding="UTF-8") as infile:
                input_dict = json.load(fp=infile)
        else:
            try:
                with open(loc, "r") as infile:
                    input_dict = json.load(fp=infile)
            except UnicodeDecodeError:
                with gzip.open(loc, "rt", encoding="UTF-8") as infile:
                    input_dict = json.load(fp=infile)
        pr = cls().from_dict(input_dict)
        return pr

    def to_markers(
        self, priorMat: pd.DataFrame, num: int = 20, index: List[str] = None
    ) -> pd.DataFrame:
        ii = self.U.columns[self.U.sum(axis=0) > 0]  # ii <- which(colsums(plierRes$U, parallel = TRUE) > 0)

        if index is not None:
            ii = np.intersect1d(ii, index)
        # if !is.null(index):
        #   ii <- intersect(ii, index)

        Zuse = self.Z.loc[:, ii]  # Zuse <- plierRes$Z[, ii, drop = F]

        # for (i in seq_along(ii)):
        #   lv <- ii[i]
        #   paths <- names(which(plierRes$U[, lv] < 0.01))
        #   genes <- names(which(rowsums(x = priorMat[, paths], parallel = TRUE) > 0))
        #   genesNotInPath <- setdiff(rownames(Zuse), genes)
        #   Zuse[genesNotInPath, i] <- 0

        for i in tqdm(ii):
            paths = self.U.index[(self.U.loc[:, i] < 0.01)].to_numpy()
            genes = priorMat[(priorMat.loc[:, paths].sum(axis=1) > 0)].index.to_numpy()
            genesNotInPath = Zuse.index[~Zuse.index.isin(genes)]
            Zuse.loc[genesNotInPath, i] = 0

        tag = Zuse.rank(ascending=False)  # tag <- apply(-Zuse, 2, rank)
        tag.columns = self.B.index[
            (self.U.sum(axis=0) > 0)
        ]  # colnames(tag) <- rownames(plierRes$B)[ii]
        iim = tag.min(axis=1)  # iim <- apply(tag, 1, min)
        iig = iim.index[iim <= num]  # iig <- which(iim <= num)
        tag = tag.loc[iig, :]  # tag <- tag[iig, ]
        iin = tag.apply(lambda x: x <= num).sum(
            axis=1
        )  # iin <- rowsums(tag <= num, parallel = TRUE)
        iimulti = iin.index[iin > 1]  # iimulti <- which(iin > 1)
        if len(iimulti) > 0:
            rprint(f"Genes not matched uniquely: {', '.join(iimulti.values)}")

        tag = tag.apply(lambda x: x <= num).astype(int)  # tag <- (tag <= num) + 1 - 1

        return tag

    # @require(lambda ngenes: ngenes > 0)
    # def getEnrichmentVals(
    #     self,
    #     pathwayMat: pd.DataFrame,
    #     ngenes: int = 50,
    #     auc_cutoff: float = 0.7,
    #     fdr_cutoff: float = 0.01,
    # ) -> pd.DataFrame:

    #     pathwayMat = pathwayMat.loc[self.Z.index, self.U.index]
    #     # TODO: okay, this isn't right, but getEnrichmentVals isn't currently used nor is it used in {PLIER}, so... ?
    #     Uuse = np.where(self.U < auc_cutoff, 0, self.U)
    #     Uuse = np.where(self.Up > getCutoff(self, fdr_cutoff), 0, self.U)
    #     intop = np.zeros(self.Z.shape[1])
    #     inpath = np.zeros(self.Z.shape[1])

    #     for i in range(intop):
    #         iipath = np.where(Uuse.iloc[:, i] > 0)
    #         if len(iipath) > 0:
    #             pathGenes = pathwayMat.loc[
    #                 pathwayMat.iloc[:, iipath].apply(sum, axis="columns") > 0, :
    #             ].index
    #             topGenes = (
    #                 self.Z.iloc[:, i].sort_values(ascending=False)[1:ngenes].index
    #             )
    #             pathGenesInt = topGenes.intersection(pathGenes)
    #             inpath[i] = len(pathGenes)
    #             intop[i] = len(pathGenesInt)

    #     return pd.DataFrame(data={1: intop / inpath, 2: intop, 3: inpath})
