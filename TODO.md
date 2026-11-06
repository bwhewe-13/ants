# To Do

## Planned Updates for 0.3.0
- [x] Implement artificial scattering for ray effect correction (2D).
- [x] Add anisotropic scattering support.
- [x] Add chi vector support.
- [ ] Add 3D transport sweep capabilities.
- [x] Add 3D interpolation capabilities.
- [x] Convert interp1d and interp2d files to Cython.
- [x] Add parallelism improvements (OpenMP tuning and/or distributed strategy).
- [x] Remove the GIL from group parallelism (`ParallelType.GROUP`/`BOTH`).
    - The group `prange` in `jacobi_iteration` (`multi_group_1d.pyx`) wraps each
      sweep in `with gil:`, so the groups run one at a time (~0.9x speedup).
    - Same pattern in `jacobi_iteration_aniso` and `multi_group_2d.pyx`.
    - The sweeps need the GIL because they allocate numpy work arrays
      (`tools.array_*`) and contain their own angle `prange`.
    - Fix: preallocate per-group buffers before the loop and add a `nogil`
      sweep entry point; then `test_multigroup_1d_speedup` should pass.
- [x] Complete spatial MMS verification for remaining unverified methods.
- [ ] Limit the memory footprint (boundary sources, time-dependent problems, etc.)
- [x] Update nearby1d and nearby2d function calls.


## Updates for 0.4.0
- [ ] Add adjoint equations (1D/2D/3D).
- [ ] Add 2D/1D projections.
- [ ] Add adaptive mesh refinement (AMR).
- [ ] Add Neutron depletion calculations.
- [ ] Add delay neutrons.
- [ ] Add neutron diffusion solve capability.
- [ ] Add Coarse Mesh Finite Difference (CMFD) Acceleration.
- [ ] Optimize the DMD implementation.


## Future Updates
- [ ] Add geometry support (gmsh, shapely, PyVista).
- [ ] Add Diffusion Synthetic Acceleration.
- [ ] Add Generalized Minimal Residual (GMRES) Acceleration.
- [ ] Add banded triangular meshes (2D).
- [ ] Add r-z geometry.
- [ ] Implement discontinuous Galerkin spatial discretization (1D/2D).
- [x] Add Step Characteristic spatial discretization support for 2D.
