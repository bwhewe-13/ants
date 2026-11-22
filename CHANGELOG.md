# Changelog

## 0.3.1 (2026-11-08)

### Added
- ANTS is on PyPI as `ants-sn` (`pip install ants-sn`). Wheels are built for
  CPython 3.11-3.14 on Linux (x86_64 and aarch64), macOS 15 or newer (arm64
  and x86_64), and Windows (x86_64), so no compiler is needed. The package is
  still imported as `ants`.

### Changed
- The distribution is named `ants-sn` (was `ants`). Run `pip uninstall ants`
  before installing `ants-sn` over an earlier source install.
- The wheels are built without `-march=native` so they run on any CPU. Source
  builds still use it and can be a little faster.

## 0.3.0 (2026-11-08)

### Added
- Three-dimensional fixed source problems with `ants.fixed3d.fixed_source`
  and `Geometry.SLAB3D`. Supports step, diamond difference, and step
  characteristic differencing, source iteration and DMD, anisotropic
  scattering, artificial scattering, the angular flux and cell edge fluxes,
  and angle and group parallelism. New inputs are `GeometryData.delta_z`,
  `GeometryData.bc_z`, and `SourceData.boundary_z`, with `angular_xyz` for
  the quadrature.
- `ants.spatial3d`, `ants.extrude3d`, and `ants.weight_spatial3d` for 3D
  medium maps. `weight_spatial2d` takes `chi` and `decimals`.
- Kobayashi 3D benchmarks (problems 1-3 with reference fluxes) in
  `ants.benchmarks.kobayashi`, and the dog leg example
  `examples/fixed_source_3d_kobayashi.py`.
- Artificial scattering (as-SN) for 2D fixed source, time-dependent, and
  hybrid problems, set with `SolverData.sigma_as` and `beta_as`. The kernel
  follows Frank et al. (2020) and uses the ordinates of the full sphere, so a
  2D problem has the same artificial scattering as a 3D problem that does
  not change in z. The outer iterations stop when the scalar flux changes by
  less than `tol_energy` and warn when they reach `max_iter_angular`.
- Anisotropic scattering in 1D and 2D for the fixed source, criticality,
  time-dependent, and nearby problems. The Legendre moments go on a last
  axis of `MaterialData.scatter`. `legendre_polynomials`,
  `spherical_harmonics`, and the related helpers are in `ants.quadrature`.
- Fission spectra with `MaterialData.chi` in all solvers.
- Step characteristic differencing in 2D.
- `CompactSource` for external and boundary sources that only take a few
  distinct values, in the 1D and 2D solvers.
- `TimeDependentData.save_to_file` writes the scalar flux of each time step
  to a memory-mapped `.npy` file instead of keeping it in memory.
- Group parallel sweeps with `SolverData.parallel` (`ParallelType.ANGLE`,
  `GROUP`, or `BOTH`). The group sweeps run without the GIL.
- Cross sections from HDF5 multigroup libraries with `ants.Library` and the
  `library` argument of `ants.materials`, with `parse_composition`,
  `create_material`, `combine_materials`, and `named_material`. Needs the
  `library` extra.
- NJOY and Serpent energy group structures (`ants.list_structures`,
  `ants.get_structure`) and the AmBe source spectrum.
- ICSBEP fast metal benchmarks (`ants.list_benchmarks`,
  `ants.get_benchmark`, `ants.run_suite`) and `examples/icsbep_suite.py`.
- `ants.utils.interp3d`. `interp1d` and `interp2d` are now written in
  Cython.
- `python -m ants.utils.mesh2d` to generate or resize 2D weight matrices.
- Warnings when an iteration reaches its maximum without converging, and an
  error for an invalid k-effective.
- Example notebooks and hosted documentation at
  https://bwhewe-13.github.io/ants/.

### Changed
- Python 3.11 or newer is required (was 3.9). Python 3.13 and 3.14 are
  supported.
- `energy_grid` returns edges in eV and in descending order for every grid
  (the 618 group grid was in MeV) and also takes structure names. The
  integer grids 361 and 618 are the SHEM-361 and LANL-618 structures, and
  87 needs the local `ants/sources/` data.
- `nearby1d` and `nearby2d` take the same dataclass inputs as the other
  solvers.
- `artificial_scatter_matrix` uses the number of ordinates on the full
  sphere (`2 N**2` for `angular_xy(N)`), which narrows the kernel for the
  same `beta_as`. `beta_as=9` gives about the old amount of smoothing. Pass
  `angle_z` for 3D sets.
- The criticality solvers start from a seeded initial flux, so runs are
  reproducible.
- The quadrature helpers moved to `ants.quadrature`. They can still be
  imported from `ants`.
- The regions in `spatial1d` can have negative coordinates.
- Time-dependent boundary sources only hold the incoming angles, so the
  angle dimension is `angles // 2` instead of `angles`.

### Fixed
- The 2D solvers gave a wrong angular flux, cell edge flux, or
  time-dependent flux when the quadrature was not ordered for the
  reflecting boundaries, for example `angular_xy(N)` without `bc_x` and
  `bc_y`. They now raise an error. The scalar flux of a fixed source or
  criticality problem does not depend on the order.
- The sign of tau in the 1D step characteristic sweep, and the radii of
  non-uniform sphere meshes.
- Step characteristic gave NaN in void cells.
- The 1D and 2D sweeps used single precision for some coefficients.
- A bug in the 1D DMD solver and a bug in `interp2d`.
