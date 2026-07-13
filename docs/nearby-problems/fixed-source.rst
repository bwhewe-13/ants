.. _sec-mnp-fixed-source:

The Method of Nearby Problems for Fixed Source Problems
=======================================================

The method of nearby problems :cite:`roy2007` works by creating a transport
problem with an analytic solution that is approximately equivalent to a
numerical solution. This transport problem is "nearby" the transport problem
that generated the numerical solution. The nearby problem and the analytical
solution can then be compared for discretization error in a manner similar to
that between the numerical estimate and an analytical or high-fidelity reference
solution.

In order to apply the method of nearby problems to the neutron transport
equation, the discrete ordinates multigroup neutron transport equation is taken
from :eq:`nte-discrete`. The steady-state fixed source neutron transport
equation is

.. math::
   :label: mnp-discrete

   \bsOmega \cdot \nabla \psi_{m, g} + \sig[g]{t} \psi_{m, g}
   = \sum_{g'=1}^{G} \sig[g' \rightarrow g]{s} \phi_{g'}
   + \chi_{g} \sum_{g'=1}^{G} \nu_{g'} \sig[g' \rightarrow g]{f} \phi_{g'}
   + q_{m,g}

with time-independent initial and boundary conditions used to solve for the
numerical angular flux :math:`\psi_{m,g}`. This is solved through source
iteration and is explained fully in the :ref:`discrete ordinates section
<sec-nte-discrete>`.

Using the numerical angular flux at each spatial cell center, curve fits
(:math:`S_{m, g}`) are created for each energy group and angle. While quintic
Hermite splines were used in the original presentation for :math:`C^3`
continuity :cite:`roy2007`, both cubic Hermite and quintic Hermite splines are
used in this analysis. Given that the solution to a transport problem can be
non-smooth at material interfaces, separate spline curve fits are created for
each material region. The splines at each material interface do not have to be
continuous. Since the polynomial for this curve fit is known analytically, it
will act as the reference solution to the nearby problem. The formulation for
the cubic and quintic Hermite splines is discussed in the :ref:`curve fit
section <sec-mnp-curve-fit>`.

To calculate the difference between the continuous curve fit and the numerical
angular flux, a continuous residual :math:`R_{m,g}` is created such that

.. math::
   :label: mnp-residual

   R_{m,g} = \bsOmega \cdot \nabla S_{m, g} + \sig[g]{t} S_{m, g}
   - \left( \sum_{g'=1}^{G} \sig[g' \rightarrow g]{s} \overline{S}_{g'}
   + \chi_{g} \sum_{g'=1}^{G} \nu_{g'} \sig[g' \rightarrow g]{f} \overline{S}_{g'}
   + q_{m,g} \right),

where :math:`\overline{S}_{g}` is the scalar curve fit flux and is calculated in
the same manner as the numerical scalar flux in :eq:`nte-scalar`. The curve fit
must be integrated over each spatial cell and associated splines to calculate
the discrete residual :math:`r_{m,g}`. The process of calculating and
discretizing the residual is addressed in the :ref:`residual section
<sec-mnp-residual>`.

The discretized residual is then added to the original external source term
:math:`q_{m, g}`, making the updated neutron transport equation

.. math::
   :label: mnp-discrete-mnp

   \bsOmega \cdot \nabla \psi_{m, g}\mnp + \sig[g]{t} \psi_{m, g}\mnp
   = \sum_{g'=1}^{G} \sig[g' \rightarrow g]{s} \phi_{g'}\mnp
   + \chi_{g} \sum_{g'=1}^{G} \nu_{g'} \sig[g' \rightarrow g]{f} \phi_{g'}\mnp
   + q_{m,g} + r_{m,g}.

The boundary condition must also be updated to agree with the curve fit at the
boundaries, such that

.. math::
   :label: mnp-boundary-condition

   \psi_{m,g} (\bx) = S_{m, g}(\bx),

although this means that the incoming boundary source can be negative. These
equations solve for the nearby angular flux :math:`\psi_{m,g}\mnp`. The nearby
angular flux and the curve fit flux are used to estimate the discretization
error when an analytical or high-fidelity solution :math:`\Psi_{m,g}` is
unknown. The assertion being made is that the relative discretization error
(RDE) between the exact solution and numerical solution is approximately
equivalent to the RDE between the curve fit flux and nearby flux,

.. math::
   :label: mnp-error-equation

   \text{RDE}_{m,g}
   = \left| \frac{\psi_{m,g} - \Psi_{m,g}}{\Psi_{m,g}} \right|
   \approx \left| \frac{\psi_{m,g}\mnp - S_{m,g}}{S_{m,g}} \right|
   = \text{MNP}_{m,g}.

The method of nearby problems does not require multiple spatial grids or
analytical solutions.

Previous work :cite:`whewell2023` has shown that the method of nearby problems
is effective with discrete ordinates multigroup neutron transport problems for
one-dimensional problems using a discrete ordinates solver. This analysis
includes results for one- and two-dimensional problems using both a discrete
ordinates solver and a Monte Carlo based solver, in this instance MC/DC
:cite:`morgan2024`. To run the method of nearby problems with a Monte Carlo
solver, the numerical flux and nearby flux calculations change. The curve fit
flux and the residual calculations use the same procedure as the
:math:`S_N` nearby problems, taking into consideration the Monte Carlo
normalization. After applying the residual as an additional source term for the
Monte Carlo solver, the same error approximation in :eq:`mnp-error-equation` is
used.
