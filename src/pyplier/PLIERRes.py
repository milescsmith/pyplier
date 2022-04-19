import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd


class PLIERResults(object):
    def __init__(
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

    def from_dict(self, source):
        if "B" in source:
            self.B = pd.DataFrame.from_dict(source["B"])
        if "Z" in source:
            self.Z = pd.DataFrame.from_dict(source["Z"])
        if "U" in source:
            self.U = pd.DataFrame.from_dict(source["U"])
        if "C" in source:
            self.C = pd.DataFrame.from_dict(source["C"])

        if "L1" in source:
            self.L1 = source["L1"]
        if "L2" in source:
            self.L2 = source["L2"]
        if "L3" in source:
            self.L3 = source["L3"]

        if "heldOutGenes" in source:
            self.heldOutGenes = source["heldOutGenes"]
        if "withPrior" in source:
            self.withPrior = source["withPrior"]

        if "residual" in source:
            self.residual = pd.DataFrame.from_dict(source["residual"])
        if "Uauc" in source:
            self.Uauc = pd.DataFrame.from_dict(source["Uauc"])
        if "Up" in source:
            self.Up = pd.DataFrame.from_dict(source["Up"])
        if "summary" in source:
            self.summary = pd.DataFrame.from_dict(source["summary"])

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
    def from_disk(cls, loc: Path):
        try:
            with open(loc, "r") as infile:
                input_dict = json.load(fp=infile)
        except UnicodeDecodeError:
            with gzip.open(loc, "rt", encoding="UTF-8") as infile:
                input_dict = json.load(fp=infile)
        pr = cls()
        pr.from_dict(input_dict)
        return pr
