# src/pyplier/__init__.py
try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

# from .num_pc import num_pc
# from .pinv_ridge import pinv_ridge
# from .solveU import solveU
# from .utils import crossprod, rowNorm, setdiff, tcrossprod

import logging

import structlog

plier_logger = structlog.get_logger()

# create logger
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)

# from .logger import setup_logging

# plier_logger = setup_logging("pyplier")


def set_verbosity(v: int = 3) -> None:
    if v == 0:
        plier_logger.setLevel(logging.CRITICAL)
    elif v == 1:
        plier_logger.setLevel(logging.ERROR)
    elif v == 2:
        plier_logger.setLevel(logging.WARNING)
    elif v == 3:
        plier_logger.setLevel(logging.INFO)
    elif v == 4:
        plier_logger.setLevel(logging.DEBUG)
