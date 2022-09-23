import gzip
import json
from collections import defaultdict
from pathlib import Path
from re import findall
from typing import Any

import h5py
import numpy as np
import pandas as pd
from loguru import logger
from rich import print as rprint
from tqdm.auto import tqdm

DATAFRAME_MEMBERS = ("b", "z", "u", "c", "uauc", "up", "summary", "residual")


# TODO: should we keep the original source data as part of this object?
# TODO: should we keep the priorMat as part of this object?  Maybe make `PLIER` a member function?
class PLIERResults:
    def __init__(
        # TODO: consider making some of these csc_matrices
        self,
        b: pd.DataFrame | None = None,
        z: pd.DataFrame | None = None,
        u: pd.DataFrame | None = None,
        c: pd.DataFrame | None = None,
        l1: float = 0.0,
        l2: float = 0.0,
        l3: float = 0.0,
        held_out_genes: dict[str, list[str]] | None = None,
        with_prior: dict[str, int] | None = None,
        uauc: pd.DataFrame | None = None,
        up: pd.DataFrame | None = None,
        summary: pd.DataFrame | None = None,
        residual: pd.DataFrame | None = None,
        # TODO: gotta think about how to implement using the hdf5 backing file
        # instead of copying everything into memory
        # backed: bool = False,
        # backing_file: Optional[Path] = None,
    ):
        self.residual = residual or pd.DataFrame
        self.b = b or pd.DataFrame
        self.z = z or pd.DataFrame
        self.u = u or pd.DataFrame
        self.c = c or pd.DataFrame
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.held_out_genes = held_out_genes or defaultdict(list)
        self.with_prior = with_prior or defaultdict(int)
        self.uauc = uauc or pd.DataFrame
        self.up = up or pd.DataFrame
        self.summary = summary or pd.DataFrame
        # self.backed = backed

    def __repr__(self) -> str:
        return (
            "\n".join(
                [
                    f"{_} : {self.__dict__[_].shape[0]} rows x {self.__dict__[_].shape[1]} columns"
                    for _ in DATAFRAME_MEMBERS
                ]
            )
            + f"held_out_genes: {len(self.held_out_genes)}\n"
            f"with_prior: {len(self.with_prior)}\n"
            f"l1 is set to {self.l1:.4f}\n"
            f"l2 is set to {self.l2:.4f}\n"
            f"l3 is set to {self.l3:.4f}"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other) -> bool:
        equal = True
        for i in DATAFRAME_MEMBERS:
            if not np.isclose(self.__dict__[i], other.__dict__[i]).all():
                logger.info(f"{i} is unequal")
                rprint(f"[red]{i}[/red] is unequal")
                equal = False
        for i in ["l1", "l2", "l3", "held_out_genes", "with_prior"]:
            if self.__dict__[i] != other.__dict__[i]:
                logger.info(f"{i} is unequal")
                rprint(f"[red]{i}[/red] is unequal")
                equal = True
        return equal

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self) -> dict[str, Any]:
        dict_repr = {_: self.__dict__[_].to_dict() for _ in DATAFRAME_MEMBERS}
        dict_repr["l1"] = self.l1
        dict_repr["l2"] = self.l2
        dict_repr["l3"] = self.l3

        dict_repr["held_out_genes"] = self.held_out_genes
        dict_repr["with_prior"] = self.with_prior
        return dict_repr

    def to_json(self, loc: Path, compress: bool = False) -> bool:
        if not isinstance(loc, Path):
            loc = Path(loc)
        if compress or loc.suffix == ".gz":
            with gzip.open(loc, "wt", encoding="UTF-8") as zipfile:
                json.dump(self.to_dict(), zipfile)
        else:
            with open(loc, "w") as jsonfile:
                json.dump(obj=self.to_dict(), fp=jsonfile)
        return True

    def to_hdf5(self, loc: Path, overwrite: bool = False) -> bool:
        if not isinstance(loc, Path):
            loc = Path(loc)

        def encode_df(h5: h5py._hl.files.File, df: pd.DataFrame, key: str):
            new_group = h5.create_group(f"{key}")
            if any(df.dtypes == "object"):
                new_group["data"] = df.to_numpy().astype(bytes)
            else:
                new_group["data"] = df.to_numpy()
            new_group["index"] = df.index.tolist()
            new_group["columns"] = df.columns.tolist()

        def encode_dict(h5: h5py._hl.files.File, dc: dict[str, list[str]], key: str, dtype=None):
            dict_group = h5.create_group(f"{key}")
            for k in dc:
                if isinstance(dc[k], list):
                    dict_group.create_dataset(k, dtype=dtype, shape=(len(dc[k]),), data=dc[k])
                else:
                    dict_group.create_dataset(k, dtype=dtype, shape=(1,), data=dc[k])

        if loc.exists() is True and not overwrite:
            logger.error("File already exists. If you wish to overwrite, please rerun with 'overwrite=True'")
        else:
            if loc.exists():
                loc.unlink()
            with h5py.File(loc, mode="a") as store:
                for df_member in (
                    "b",
                    "z",
                    "u",
                    "c",
                    "uauc",
                    "up",
                    "summary",
                    "residual",
                ):
                    encode_df(h5=store, df=self[df_member], key=df_member)

                store.create_dataset("l1", (1,), dtype=float, data=self.l1)
                store.create_dataset("l2", (1,), dtype=float, data=self.l2)
                store.create_dataset("l3", (1,), dtype=float, data=self.l3)

                encode_dict(h5=store, dc=self.held_out_genes, key="held_out_enes")
                encode_dict(h5=store, dc=self.with_prior, key="with_prior", dtype=int)

    @classmethod
    def from_dict(cls, source):
        pr = cls()
        if "b" in source:
            pr.b = pd.DataFrame.from_dict(source["b"])
        if "z" in source:
            pr.z = pd.DataFrame.from_dict(source["z"])
        if "u" in source:
            pr.u = pd.DataFrame.from_dict(source["u"])
        if "c" in source:
            pr.c = pd.DataFrame.from_dict(source["c"])

        if "l1" in source:
            pr.l1 = source["l1"]
        if "l2" in source:
            pr.l2 = source["l2"]
        if "l3" in source:
            pr.l3 = source["l3"]

        if "held_out_genes" in source:
            pr.held_out_genes = source["held_out_genes"]
        if "with_prior" in source:
            pr.with_prior = source["with_prior"]

        if "residual" in source:
            pr.residual = pd.DataFrame.from_dict(source["residual"])
        if "uauc" in source:
            pr.uauc = pd.DataFrame.from_dict(source["uauc"])
        if "up" in source:
            pr.up = pd.DataFrame.from_dict(source["up"])
        if "summary" in source:
            pr.summary = pd.DataFrame.from_dict(source["summary"])
        return pr

    @classmethod
    def read_json(cls, loc: Path):
        """TODO: switch to using something like HDF5 or parquet for storage"""
        if str(loc).endswith(".gz"):
            with gzip.open(loc, "rt", encoding="UTF-8") as infile:
                input_dict = json.load(fp=infile)
        else:
            try:
                with open(loc) as infile:
                    input_dict = json.load(fp=infile)
            except UnicodeDecodeError:
                with gzip.open(loc, "rt", encoding="UTF-8") as infile:
                    input_dict = json.load(fp=infile)
        return cls().from_dict(input_dict)

    @classmethod
    def read_hdf5(cls, loc: Path, codec: str = "UTF-8"):
        def decode_df(h5: h5py._hl.files.File, group: str, codec: str = codec) -> pd.DataFrame:
            df = pd.DataFrame(
                data=h5[group]["data"],
                index=pd.Series(h5[group]["index"]).str.decode(codec),
                columns=pd.Series(h5[group]["columns"]).str.decode(codec),
            ).apply(lambda x: x.str.decode(codec) if x.dtype == "object" else x)
            return df

        def decode_dict(h5: h5py._hl.files.File, group: str) -> dict[str, list[str]]:
            decoded_dict = {
                k: (
                    []
                    if len(h5[group][k]) == 0
                    else (
                        h5[group][k][0]
                        if np.issubdtype(h5[group][k].dtype, np.number)
                        else np.char.array(h5[group][k]).decode().tolist()
                    )
                )
                for k in h5[group].keys()
            }
            return decoded_dict

        def only_int(x):
            return int(findall(r"[0-9]+", x)[0])

        try:
            with h5py.File(loc, mode="r") as store:
                with_prior = decode_dict(store, "with_prior")

                pr = cls(
                    b=decode_df(store, "b"),
                    z=decode_df(store, "z"),
                    u=decode_df(store, "u"),
                    c=decode_df(store, "c"),
                    l1=store["l1"][0],
                    l2=store["l2"][0],
                    k3=store["l3"][0],
                    held_out_genes=decode_dict(store, "held_out_genes"),
                    with_prior={k: with_prior[k] for k in sorted(with_prior, key=only_int)},
                    uauc=decode_df(store, "uauc"),
                    up=decode_df(store, "up"),
                    summary=decode_df(store, "summary")
                    .apply(lambda x: x.str.decode("UTF-8"))
                    .astype({"LV index": str, "AUC": float, "p-value": float, "FDR": float}),
                    residual=decode_df(store, "residual"),
                )
            return pr
        except FileNotFoundError:
            logger.exception(f"no such file was found at {loc!s}")

    def to_markers(
        self,
        prior_mat: pd.DataFrame,
        num: int = 20,
        index: list[str] | None = None,
        persistent_progress: bool = True,
        disable_progress: bool = False,
    ) -> pd.DataFrame:
        ii = self.u.columns[self.u.sum(axis=0) > 0]  # ii <- which(colsums(plierRes$U, parallel = TRUE) > 0)

        if index is not None:
            ii = np.intersect1d(ii, index)
        # if !is.null(index):
        #   ii <- intersect(ii, index)

        zuse = self.z.loc[:, ii]  # zuse <- plierRes$Z[, ii, drop = F]

        # for (i in seq_along(ii)):
        #   lv <- ii[i]
        #   paths <- names(which(plierRes$U[, lv] < 0.01))
        #   genes <- names(which(rowsums(x = priorMat[, paths], parallel = TRUE) > 0))
        #   genesNotInPath <- setdiff(rownames(zuse), genes)
        #   zuse[genesNotInPath, i] <- 0

        for i in tqdm(ii, disable=disable_progress, leave=persistent_progress, desc="Converting markers"):
            paths = self.u.index[(self.u.loc[:, i] < 0.01)].to_numpy()
            genes = prior_mat[(prior_mat.loc[:, paths].sum(axis=1) > 0)].index.to_numpy()
            genes_not_in_path = zuse.index[~zuse.index.isin(genes)]
            zuse.loc[genes_not_in_path, i] = 0

        tag = zuse.rank(ascending=False)  # tag <- apply(-zuse, 2, rank)
        tag.columns = self.b.index[(self.u.sum(axis=0) > 0)]  # colnames(tag) <- rownames(plierRes$B)[ii]
        iim = tag.min(axis=1)  # iim <- apply(tag, 1, min)
        iig = iim.index[iim <= num]  # iig <- which(iim <= num)
        tag = tag.loc[iig, :]  # tag <- tag[iig, ]
        iin = tag.apply(lambda x: x <= num).sum(axis=1)  # iin <- rowsums(tag <= num, parallel = TRUE)
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
