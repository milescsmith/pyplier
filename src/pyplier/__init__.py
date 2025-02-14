# src/pyplier/__init__.py
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


# from pyplier.logging import init_logger
from pyplier.pathways import combine_paths, pathway_from_gmt
from pyplier.plier import PLIER
from pyplier.plier_res import PLIERResults

# def set_verbosity(v: int = 3) -> None:
#     init_logger(v)


__all__ = [
    "combine_paths",
    "pathway_from_gmt",
    "PLIER",
    "PLIERResults",
]
