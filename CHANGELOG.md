# Changelog

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

[0.10.0]: https://github.com/milescsmith/pyplier/releases/compare/0.11.0..0.12.0
[0.10.0]: https://github.com/milescsmith/pyplier/releases/compare/0.10.0..0.11.0
[0.9.0]: https://github.com/milescsmith/pyplier/releases/compare/0.9.0..0.10.0
[0.9.0]: https://github.com/milescsmith/pyplier/releases/compare/0.8.0..0.9.0
[0.8.0]: https://github.com/milescsmith/pyplier/releases/compare/0.7.0..0.8.0
[0.7.0]: https://github.com/milescsmith/pyplier/releases/compare/0.6.2..0.7.0
[0.6.2]: https://github.com/milescsmith/pyplier/releases/compare/0.6.1..0.6.2
[0.6.i]: https://github.com/milescsmith/pyplier/releases/compare/0.6.0..0.6.1
[0.6.0]: https://github.com/milescsmith/pyplier/releases/tag/0.6.0
