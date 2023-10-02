
# A Neutron Transport Solution (ANTS)

A Neutron Transport Solution (ANTS) calculates the neutron flux for both criticality and fixed source problems of one dimensional slabs and spheres and two dimensional slabs using the discrete ordinates method and written in Cython. 

There are a number of different acceleration methods used including a collision-based hybrid method, machine learning DJINN models to predict matrix-vector multiplication, dynamic mode decomposition, and synthetic diffusion acceleration (DSA).

There are also verification procedures to ensure both the code and solutions are correct. For code verification, manufactured solutions are used for one- and two-dimenisonal slab problems to ensure proper discretization. Solution verification uses the method of nearby problems, which uses one spatial grid. 

&nbsp;

## One Dimensional Features
| Spatial Discretization    | Temporal Discretization    | Multigroup Solve          | K-Eigenvalue Solve      |
|---------------------------|----------------------------|---------------------------|-------------------------|
| &#9745; Step Method       | &#9745; Backward Euler     | &#9745; Source Iteration  | &#9745; Power Iteration |
| &#9745; Diamond Difference    | &#9745; BDF2           | &#9744; DSA               | &#9744; DJINN           |
| &#9744; Step Characteristic   | &#9745; Crank-Nicolson | &#9744; CMFD              | &#9744; DMD             |
| &#9744; Discontinuous Galerkin| &#9745; TR - BDF2      | &#9744; GMRES             | &#9744; Davidson Method |

&nbsp;

## Two Dimensional Features
| Spatial Discretization    | Temporal Discretization    | Multigroup Solve          | K-Eigenvalue Solve      |
|---------------------------|----------------------------|---------------------------|-------------------------|
| &#9745; Step Method       | &#9745; Backward Euler     | &#9745; Source Iteration  | &#9745; Power Iteration |
| &#9745; Diamond Difference    | &#9745; BDF2           | &#9744; DSA               | &#9744; DMD             |
| &#9744; Step Characteristic   | &#9745; Crank-Nicolson | &#9744; CMFD              | &#9744; Davidson Method |
| &#9744; Discontinuous Galerkin| &#9745; TR - BDF2      | &#9744; GMRES             |                         |

&nbsp;

## Verification
- [ ] Method of Manufactured Solutions (MMS)
- [ ] Method of Nearby Problems (MNP)

&nbsp;

### To Do 
- [ ] 1D / 2D Diffusion Equation
- [ ] Correct for Ray Effects (2D)
