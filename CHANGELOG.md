# Changelog

## [1.7.0] - 2022-09-21

### Changed

- replaced `pyplier.utils.rowNorm` with `pyplier.utils.zscore` as that is what it was anyway
  - also, it is no self applies - you will need to run it over rows manually (i.e. df.apply(zscore, axis=1))
- removed `pyplier.utils.colNorm` since it was unused

### Fixed

- `pyplier.utils.zscore` *should* be able to handle rows where all the values are the same (return 0, since all
  values are right on the standard deviation of 0)

## [1.6.0] - 2022-09-21

### Fixed

- fixed an issue when reading a PLIERResult object stored as an h5 has empty lists as part of the "heldOutGenes" member

## [1.5.5] - 2022-09-13

### Fixed

- saving non-integer numeric dictionaries in PLIERResults.withPrior to hdf5 no
  longer results in their being inappropriately cast to integers

## [1.5.4] - 2022-09-13

### Added

- Automated unit testing on Github

### Changed

- Instead of using the stale civisanalytics/python-glmnet, switch to the slightly-less stale replicahq/python-glmnet
  (which uses pep518-compliant install, obviating any issues with needing numpy installed prior to glmnet)

## [1.5.3] - 2022-09-12

### Fixed

- Another attempt at fixing an issue where a PLIERResults object could not be saved as hdf5 because of the "LV index" text column in the
  summary dataframe (and this is why you should write unit tests prior to saying you "fixed" something)

## [1.5.2] - 2022-09-12

### Fixed

- Fixed an issue where a PLIERResults object could not be saved as hdf5 because of the "LV index" text column in the
  summary dataframe


## [1.5.1] - 2022-09-12

### Fixed

- Removed stray `glmnet-py` dep

## [1.5.0] - 2022-09-09

### Added

- Added methods to save and read PLIERResults objects to disk using h5py, which should be dramatically faster than using
  json
- Added a `__getitem__` method to the `PLIERResults` class

### Changed

- Renamed `PLIERResults.to_disk()` to `PLIERResults.to_json()` and `PLIERResults.from_dist()` to `PLIERResults.read_json()`


## [1.4.0] - 2022-09-02

### Fixed

- Better handling for NaNs and Inf values in the normalized data when running `plier()`
- Explictly list submodules as part of `__all__` in `__init__.py` to hopefully solve certain submodules
  not being available when importing the main module

### Misc

- Update dependencies
- Cleanup testing data

## [1.3.0] - 2022-08-22

### Changed

- revert newer (3.9+) style typing soas to reenable Python 3.8

## [1.2.0] - 2022-08-22

### Changed

- Simplified `solveU`

## [1.1.1] - 2022-08-17

### Fixed

- Should probably actually try using `pyplier.plotting.plotTopZ` before pushing.

## [1.1.0] - 2022-08-17

### Fixed

- `pyplier.plotting.plotTopZ` actually works now.

## [1.0.1] - 2022-08-15

### Fixed

- Replaced `rich.console.Console` with just using `rich.print` so as to avoid circular imports

## [1.0.0] - 2022-08-15

### Changed

- Reorganized the submodules to make a little more sense and so that not everything was 
  `plier.FUNCTION.FUNCTION`; some functions changed to be PLIERRes methods

### Removed

- Removed multiple functions


## [0.14.0] - 2022-08-15

### Fixed

- In `nameB`, handle the situation where no pathways have a FDR less than the cutoff

## [0.13.1] - 2022-08-07

## Removed

- Removed a few unused or unnecessary dependencies to reduce the exposure to
  vulnerabilities

## [0.13.0] - 2022-07-31

### Added

- Unit test for `pyplier.combinePaths.combinePaths()` 
  - also, while the results do not match those from R {PLIER}, pyplier's
  version is technically correct  - PLIER's version has a bug

### Fixed

- Replaced inappropriate use of `numpy.exp()` where I should have used `numpy.power()`
in `pyplier.AUC.mannwhitney_conf_int()`
- remove stupid `from typing import dict, list` leftovers
- `pyplier.combinePaths.combinePaths()` now returns an array of ints


## [0.12.0] - 2022-07-17

- All apparently used functions implemented

### Added

- `plotU()` & `VarianceExplained()`

### Changed

- Changed minimum required Python version to 3.9

## [0.11.0] - 2022-07-04

### Added

- `plotTopZallPath()`

### Fixed

- `plotTopZ()` and `solveU()` now properly use PLIERRes members

## [0.10.0] - 2022-06-26

### Added

- `plierResToMarkers`, `plotMat`, and `plotTopZ`

## [0.9.0] - 2022-06-24

### Added

- Unit tests for the `PLIERResults` class
- Start add use of `structlog`
- added `__eq__()` method to the `PLIERResults` class

### Changed

- dev container now uses python 3.9
- Replaced a few math functions in `pyplier.AUC.mannwhitneyu_conf_int()` 
with their numpy equivalents
- `pyplier.crossVal.crossVal()` now sets the "pathway" column as the index for
the `summary` dataframe
- changed `pyplier.PLIERResults.from_dict()` method to a `classmethod`
- gzipped several of the testing files

### Fixed

- Fixed column names for testing dataframes in `test_solveU`

## [0.8.0] - 2022-04-19

### Added

- New version of `mannwhitney_conf_u()`, which is a direct translation of the R
code used in the portion of `stats::wilcox.test()` that generates the
confidence interval for `PLIER::AUC()`
- `__repr__()` method to the PLIERRes class

### Changed

- Simpler version of `solveU()` (at least temporarily)
- Started using typechecked
- Overhaul `crossVal()`

### Fixes

- `plier()` now runs and appears to do so essentially correctly; however,
at current, it reports fewer identified LVs than {PLIER} does (about half
as many)

### Removed

- Got rid of `commonRows()` as it was unnecessary

## [0.7.0] - 2022-02-08

### Added

- First draft of the main PLIER function, though it 


### Changed

- Compress test data and change tests to reflect
- Minor code changes to eliminate pytest warnings
- Switched `PLIERResults` from a TypedDict to a class

## [0.6.2] - 2022-02-07

### Changed

- Removed numba as a dependency for now (it wasn't being used anyway), allowing
the required numpy version to be updated to >1.21 (lower version had a vulnerability)
- Update all dependencies

## [0.6.1] - 2022-02-06

### Changed

- Update dependenciess to eliminate ipython vulnerability

## [0.6.0] - 2022-02-06

### Added

- `solveU()` with unit test (though, no hypothesis-based unit tests yet)
- A Docker devcontainer
- Import functions into __init__ to simplify intramodule use
- Add icontract-based function input validation to some functions


[1.5.3]: https://github.com/milescsmith/pyplier/releases/compare/1.5.2..1.5.3
[1.5.2]: https://github.com/milescsmith/pyplier/releases/compare/1.5.1..1.5.2
[1.5.1]: https://github.com/milescsmith/pyplier/releases/compare/1.5.0..1.5.1
[1.5.0]: https://github.com/milescsmith/pyplier/releases/compare/1.4.0..1.5.0
[1.4.0]: https://github.com/milescsmith/pyplier/releases/compare/1.3.0..1.4.0
[1.3.0]: https://github.com/milescsmith/pyplier/releases/compare/1.2.0..1.3.0
[1.2.0]: https://github.com/milescsmith/pyplier/releases/compare/1.1.1..1.2.0
[1.1.1]: https://github.com/milescsmith/pyplier/releases/compare/1.1.0..1.1.1
[1.1.0]: https://github.com/milescsmith/pyplier/releases/compare/1.0.1..1.1.0
[1.0.1]: https://github.com/milescsmith/pyplier/releases/compare/1.0.0..1.0.1
[1.0.0]: https://github.com/milescsmith/pyplier/releases/compare/0.14.0..1.0.0
[0.14.0]: https://github.com/milescsmith/pyplier/releases/compare/0.13.1..0.14.0
[0.13.1]: https://github.com/milescsmith/pyplier/releases/compare/0.13.0..0.13.1
[0.13.0]: https://github.com/milescsmith/pyplier/releases/compare/0.12.0..0.13.0
[0.12.0]: https://github.com/milescsmith/pyplier/releases/compare/0.11.0..0.12.0
[0.11.0]: https://github.com/milescsmith/pyplier/releases/compare/0.10.0..0.11.0
[0.10.0]: https://github.com/milescsmith/pyplier/releases/compare/0.9.0..0.10.0
[0.9.0]: https://github.com/milescsmith/pyplier/releases/compare/0.8.0..0.9.0
[0.8.0]: https://github.com/milescsmith/pyplier/releases/compare/0.7.0..0.8.0
[0.7.0]: https://github.com/milescsmith/pyplier/releases/compare/0.6.2..0.7.0
[0.6.2]: https://github.com/milescsmith/pyplier/releases/compare/0.6.1..0.6.2
[0.6.1]: https://github.com/milescsmith/pyplier/releases/compare/0.6.0..0.6.1
[0.6.0]: https://github.com/milescsmith/pyplier/releases/tag/0.6.0
