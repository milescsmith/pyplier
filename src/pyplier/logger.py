# stolen wholesale from `anndata <https://github.com/theislab/anndata/blob/master/anndata/_core/anndata.py>`_
# BSD 3-Clause License

# Copyright (c) 2017-2018 P. Angerer, F. Alexander Wolf, Theis Lab
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# import logging
# import os

# _previous_memory_usage = None
# cupcake_logger.addHandler(logging.StreamHandler())  # Logs go to stderr
# cupcake_logger.handlers[-1].setFormatter(logging.Formatter("%(message)s"))
# cupcake_logger.handlers[-1].setLevel("INFO")


# def get_logger(name):
#     """\
#     Creates a child logger that delegates to cupcake_logger
#     instead to logging.root
#     """
#     return cupcake_logger.manager.getLogger(name)


import logging
from typing import Optional

import coloredlogs


def setup_logging(name: Optional[str] = None, level="DEBUG") -> logging.Logger:
    coloredlogs.DEFAULT_FIELD_STYLES = {
        "asctime": {"color": "green"},
        "levelname": {"bold": True, "color": "red"},
        "module": {"color": 73},
        "funcName": {"color": 74},
        "lineno": {"bold": True, "color": "green"},
        "message": {"color": "yellow"},
    }

    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    logger.propagate = True

    if level == "DEBUG":
        logger_level = logging.DEBUG
    elif level == "INFO":
        logger_level = logging.INFO
    elif level == "WARNING":
        logger_level = logging.WARNING
    elif level == "ERROR":
        logger_level = logging.ERROR
    elif level == "CRITICAL":
        logger_level = logging.CRITICAL

    logger.setLevel(logger_level)
    coloredlogs.install(
        level=level,
        fmt="[%(funcName)s(%(lineno)d)]: %(message)s",
        logger=logger,
    )

    if name:
        fh = logging.FileHandler(f"{name}.log")
    else:
        fh = logging.FileHandler(f"{__name__}")
    formatter = logging.Formatter(
        "[%(asctime)s] {%(module)s:%(funcName)s():%(lineno)d} %(levelname)s - %(message)s"
    )
    fh.setLevel(logger_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
