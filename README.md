
# Accelerated Neutron Transport Solution (ANTS)

Accelerated Neutron Transport Solution (ANTS) calculates the neutron flux for both criticality and fixed source problems of one dimensional slabs and spheres and two dimensional slabs using the discrete ordinates method and written in Cython. 

There are a number of different acceleration methods used including a collision-based hybrid method, machine learning DJINN models to predict matrix-vector multiplication, dynamic mode decomposition, and synthetic diffusion acceleration (DSA).

There are also verification procedures to ensure both the code and solutions are correct. For code verification, manufactured solutions are used for one- and two-dimenisonal slab problems to ensure proper discretization. Solution verification uses the method of nearby problems, which uses one spatial grid. 


### Spatial Discretization
- [ ] One Dimensional
	- [x] Step Method
	- [x] Diamond Difference
	- [ ] Step Characteristic Method	
	- [ ] Discontinuous Galerkin
- [ ] Two Dimensional
	- [x] Square Diamond Difference
	- [x] Square Step Method
	- [ ] Square Step Characteristic

### Temporal Discretization
- [x] Backward Euler (BDF1)
- [ ] Backward differentiation formula 2 (BDF2)
- [ ] Trapezoidal Rule BDF2 (TR-BDF2)

### Solution Techniques
- [x] Source Iteration
- [x] Power Iteration
- [ ] Machine Learning (DJINN)
- [ ] Machine Learning (Auto-DJINN)
- [ ] Collision-Based Hybrid Method
- [ ] Dynamic Mode Decomposition (DMD)
- [ ] Synthetic Diffusion Acceleration (DSA)
- [ ] Generalized Minimal Residual Method (GMRES)
- [ ] Coarse Mesh Finite Difference (CMFD)

### Verification
- [ ] Method of Manufactured Solutions (MMS)
- [ ] Method of Nearby Problems (MNP)
