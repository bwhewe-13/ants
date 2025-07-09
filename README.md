
# A Neutron Transport Solution (ANTS)

A Neutron Transport Solution (ANTS) calculates the neutron flux for both criticality and fixed source problems of one dimensional slabs and spheres and two dimensional slabs using the discrete ordinates method and written in Cython. 

There are a number of different acceleration methods used including a collision-based hybrid method, machine learning models to predict matrix-vector multiplication, dynamic mode decomposition, and synthetic diffusion acceleration (DSA).

There are also verification procedures to ensure both the code and solutions are correct. For code verification, manufactured solutions are used for one- and two-dimenisonal slab problems to ensure proper discretization. Solution verification uses the method of nearby problems, which uses one spatial grid. 

&nbsp;

## One Dimensional Features
| Spatial Discretization    | Temporal Discretization    | Multigroup Solve          | K-Eigenvalue Solve      |
|---------------------------|----------------------------|---------------------------|-------------------------|
| &#9745; Step Method       | &#9745; Backward Euler     | &#9745; Source Iteration  | &#9745; Power Iteration |
| &#9745; Diamond Difference    | &#9745; BDF2           | &#9744; DSA               | &#9744; DJINN           |
| &#9745; Step Characteristic   | &#9745; Crank-Nicolson | &#9745; DMD               | &#9745; DMD             |
| &#9744; Discontinuous Galerkin| &#9745; TR - BDF2      | &#9744; GMRES             | &#9744; Davidson Method |

&nbsp;

## Two Dimensional Features
| Spatial Discretization    | Temporal Discretization    | Multigroup Solve          | K-Eigenvalue Solve      |
|---------------------------|----------------------------|---------------------------|-------------------------|
| &#9745; Step Method       | &#9745; Backward Euler     | &#9745; Source Iteration  | &#9745; Power Iteration |
| &#9745; Diamond Difference    | &#9745; BDF2           | &#9744; DSA               | &#9745; DMD             |
| &#9744; Step Characteristic   | &#9745; Crank-Nicolson | &#9745; DMD               | &#9744; Davidson Method |
| &#9744; Discontinuous Galerkin| &#9745; TR - BDF2      | &#9744; GMRES             |                         |

&nbsp;

## Code and Solution Verification
- &#9744; Spatial Method of Manufactured Solutions (1D/2D)
    - &#9745; &#9745; Step Method
    - &#9745; &#9745; Diamond Difference
    - &#9745; &#9744; Step Characteristics
    - &#9744; &#9744; Discontinuous Galerkin
- &#9745; Temporal Method of Manufactured Solutions
    - &#9745; &#9745; Backward Euler
    - &#9745; &#9745; BDF2
    - &#9745; &#9745; Crank-Nicolson
    - &#9745; &#9745; TR - BDF2
- &#9745; &#9745; Criticality Benchmarks \
    (Analytical Benchmark Test Set for Criticality Code Verification, Sood et al) 
- &#9745; &#9745; Method of Nearby Problems (MNP)

&nbsp;

## Features To Add
- &#9744; Ray Effect Corrections (2D)
- &#9744; Adjoint Equations (1D/2D)
- &#9744; Acceleration Techniques (DSA, GMRES, CMFD)
- &#9744; Optimize DMD Implementation
- &#9744; Banded Triangular Meshes (2D)
