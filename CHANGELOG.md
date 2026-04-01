# Changelog

All notable changes to this project are documented in this file.

## v0.6.0

- Replaced hardcoded X-ray and neutron scattering tables with `periodictable`.
- Removed packaged CSV scattering data and the old table-backed helper APIs.
- Added `ScatteringComposition` as a composition-only helper for scattering weights and prefactors.
- Tightened scattering input validation and improved class naming and docstrings.
- Updated package metadata and dependencies for the new scattering backend.

## v0.5.8

- Added NOMAD prefactor calculation.
- Fixed neutron scattering length table import.
- Bumped version to `v0.5.8`.

## v0.5.7

- Added Lorch and zero padding support to the X-ray cutoff path for `g(r)`.
- Fixed `weight_RDF_for_scattering` X-ray weighting behavior.
- Improved fitting utilities, including `fitPeak` parameter input handling and lmfit float/array compatibility.
- Added skew parameter support.

## v0.5.6

- Fixed weight input handling.
- Fixed keyword handling.
