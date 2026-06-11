.. _chapter-mnp:

The Method of Nearby Problems
=============================

The neutron transport equation is solved numerically through both deterministic
(e.g. discrete ordinates) and probabilistic (e.g. Monte Carlo) methods
:cite:`lewis1993`. Two types of verification are used to ensure that these
simulations are approximating the solution correctly. The first type is code
verification :cite:`roy2005`. This is a process to ensure that the numerical
discretization used by computer codes is functioning properly. One such
technique is the method of manufactured solutions (MMS), which starts with an
analytical solution and works backwards to solve for the initial and boundary
conditions of the problem. The initial and boundary conditions are used to
create a numerical approximation to the known "manufactured" solution. Several
numerical approximations with different mesh sizes are used to estimate the
discretization error and verify the correct rate of convergence. To prevent
requiring significant amounts of time to formulate these analytical solutions
and their associated initial and boundary conditions, simple, and often
nonphysical, solutions are used. This technique has been implemented numerous
times with neutron transport solvers :cite:`wang2019`.

The second type of verification is solution verification. This is a process to
ensure that the discretization error associated with a given numerical solution
is sufficiently close to the true solution of the governing equation
:cite:`roy2005`. As opposed to the simple solutions from the method of
manufactured solutions, solution verification is often used with complex
problems and is implemented when providing engineering analysis. In most
instances, the true solution to the governing equation is neither known nor
readily available. For these cases, techniques such as extrapolation-based error
estimation are used. One extrapolation-based error estimation employs Richardson
extrapolation to approximate a higher-order numerical solution to estimate the
correct discretization error :cite:`roy2005`. This method is effective at
determining the global error for a problem, but requires at least two (high- and
low-fidelity) grids. In terms of estimating spatial error, which this chapter
focuses on, using multiple spatial grids can become difficult in instances with
unstructured meshes. The ability to develop separate high- and low-fidelity
unstructured grids might not be obtainable and can require significant
computational and analyst time :cite:`mavriplis1997`.

Rather than requiring multiple spatial grids, the method of nearby problems
:cite:`roy2007` uses single-grid error estimation for solution verification.
After solving for a numerical solution, an analytical function can be
constructed by interpolating with a known polynomial function. The residual
between the analytical solution and the numerical solution is calculated and
added to the original problem as an additional source term. Utilizing the
additional source term on the same spatial grid calculates the nearby solution.
The nearby solution can be compared to the analytical solution and is used to
estimate the discretization error. Additional measures must be taken to
calculate the spatial discretization error with the method of nearby problems
for :math:`k`-eigenvalue problems. This technique has been proven to work with
various PDE problems :cite:`roy2007` and expands upon previous work
:cite:`whewell2023` as it applies to the neutron transport equation.

This chapter is organized as follows. The :ref:`fixed-source formulation
<sec-mnp-fixed-source>` explains the method of nearby problems as it pertains to
fixed-source problems and how it is directly applied to the multigroup neutron
transport equation. The :ref:`k-eigenvalue formulation <sec-mnp-criticality>`
discusses the application of the method to :math:`k`-eigenvalue problems. The
:ref:`Hermite spline curve fit <sec-mnp-curve-fit>` and :ref:`residual
formulation <sec-mnp-residual>` then detail the construction of the curve-fit
flux and the discretization of the residual introduced in the fixed-source
formulation.

.. toctree::
   :maxdepth: 2

   fixed-source.rst
   criticality.rst
   curve-fit.rst
   residual.rst
