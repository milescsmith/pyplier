r"""*Single-source version number for* ``azimuth``.
``pyplier``: A Python translation of the R package {PLIER}
**Authors**
    Miles Smith
**Source Repository**
    http://www.github.com/milescsmith/pyplier
**Documentation**
    See README.md at the GitHub repository
**License**
    GNU Public License, v2; see |license_md|_ for full license terms
"""

from importlib.metadata import metadata, version

try:
    __author__ = metadata(__name__)["Author"]
except KeyError:
    __author__ = "unknown"

try:
    __email__ = metadata(__name__)["Author-email"]
except KeyError:  # pragma: no cover
    __email__ = "unknown"

try:
    __version__ = version(__name__)
except KeyError:  # pragma: no cover
    __version__ = "unknown"