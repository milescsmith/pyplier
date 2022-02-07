# Changelog

## [0.6.2] - 2022-02-07

### Changed

- Removed numba as a dependency for now (it wasn't being used anyway), allowing
the required numpy version to be updated to >1.21 (lower version had a vulnerability)
- Update all dependencies

## [0.6.1] - 2022-02-06

### Changed

- Update dependencys to eliminate ipython vulnerability

## [0.6.0] - 2022-02-06

### Added

- `solveU()` with unit test (though, no hypothesis-based unit tests yet)
- A Docker devcontainer
- Import functions into __init__ to simplify intramodule use
- Add icontract-based function input validation to some functions


[0.6.2]: https://github.com/olivierlacan/keep-a-changelog/releases/compare/0.6.1..0.6.2
[0.6.i]: https://github.com/olivierlacan/keep-a-changelog/releases/compare/0.6.0..0.6.1
[0.6.0]: https://github.com/olivierlacan/keep-a-changelog/releases/tag/0.6.0
