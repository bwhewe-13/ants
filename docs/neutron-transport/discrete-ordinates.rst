.. _sec-nte-discrete:

The Multigroup Discrete Ordinates Method
========================================

The angular component of the neutron transport equation can be discretized in a
number of different ways. The discrete ordinates (:math:`S_N`)
method :cite:`chandrasekhar2013` solves for the angular flux at specific angles
by using quadrature sets to estimate the integral over the angle. For the
angular discretization, let :math:`\bsOmega_m` and :math:`w_m`, where
:math:`m \in \cM := \{1, \ldots, M \}`, be discrete angles and weights for a
quadrature rule over the sphere: for any integrable function :math:`u` defined
point-wise everywhere on :math:`\bbS^2`,

.. math::

   \frac{1}{4 \pi} \int_{\bbS^2} d \bsOmega\, u(\bsOmega) \approx \sum_{m=1}^M w_m u(\bsOmega_m).

Angles in the :math:`x` direction are represented as :math:`\mu_{m}` and angles
in the :math:`y` direction are represented as :math:`\eta_{m}`. There are a
number of different quadrature sets that can be used for both one- and
two-dimensional problems :cite:`lewis1993,jarrell2011`. For one-dimensional
problems, the Gauss-Legendre quadrature is used and is symmetrical around
:math:`\mu = 0`. For two-dimensional problems, the product quadrature set is used
with the Gauss-Legendre and Gauss-Chebyshev quadratures. With one-dimensional
problems, the number of discrete ordinates equate to the :math:`M` number of
distinct angular directions while two-dimensional problems result in
:math:`M^2` number of distinct directions. It should also be noted that an even
number of discrete ordinates are chosen in both instances.

An anomaly that can arise with the discrete ordinates method in two dimensions is
the artificial creation of ray effects. These ray effects are oscillations with
the flux and are seen in problems with fixed point sources and low scattering
materials :cite:`lewis1993`. This is due to the inability of the quadrature
formula from correctly estimating the scalar flux from the angular flux. While
research has been done to mitigate these problems :cite:`frank2020`, a common
solution is to increase the number of discrete ordinates :cite:`lewis1993`.
Increasing the number of discrete ordinates is how ray effects are minimized in
this analysis, but ray effect corrections should be examined in future work.

The energy component is also important to discretize, as the neutron cross
sections are functions of energy :cite:`lewis1993`. While there are instances
with Monte Carlo solvers that use continuous energy, the multigroup approach is
more feasible in terms of memory and computational time :cite:`deng2019` and will
be used with this analysis. The multigroup method approximates the energy
functions as piece-wise constants in energy "bins." The energy group is
represented with :math:`g \in \cG := \{1, \ldots, G\}`, where
:math:`0 = E_{0} < E_{1} < \cdots E_{g} < E_{g+1} < \cdots < E_{G} = E_{\rm{max}}`
are the energy values. The energy bin widths are
:math:`\Delta E_{g} = E_{g} - E_{g-1}`. For the multigroup scattering cross
sections, the notation :math:`g' \to g` denotes scattering from energy group
:math:`E_{g'}` to :math:`E_g`. Downscattering (:math:`E_{g'} > E_{g}`) is seen as
a lower triangular matrix, upscattering (:math:`E_{g'} < E_{g}`) as the upper
triangular, and self-scattering (:math:`E_{g'} = E_{g}`) as the diagonal matrix.

An issue with the multigroup method is that it is often difficult to estimate the
cross sections energy functions and are exacerbated by neutron resonances, which
are high and narrow spikes in cross section reactions :cite:`rinard1991`. It is
imperative when converting continuous to multigroup energy to ensure that the
energy bins will properly characterize the cross section functions. The
quantities :math:`\sig[g]{t}`, :math:`\sig[g' \rightarrow g]{s}`,
:math:`\chi_{g}`, :math:`\nu_{g'}`, and :math:`\sig[g']{f}` are all approximate
weighted averages of their continuum counterparts. For example,

.. math::

   \sig[g' \rightarrow g]{s}(x, t) \approx \frac{\displaystyle\int_{E_{g-1}}^{E_{g}}  \displaystyle\int_{E_{g'-1}}^{E_{g'}} dE dE' \, \sig{s} (\bx, E' \rightarrow E, t) \Phi (\bx, E', t)}{\displaystyle\int_{E_{g'-1}}^{E_{g'}} dE' \, \Phi(\bx, E', t)}

and

.. math::

   \chi_{g}(x) \approx \frac{\displaystyle\int_{E_{g-1}}^{E_{g}} dE \,  \chi(\bx, E) \Phi (\bx, E, t)}{\displaystyle\int_{E_{g-1}}^{E_{g}} dE \, \Phi(\bx, E, t)}.

Cross section approximations come from the fact that :math:`\Phi(\bx, E, t)` is
not known a priori and an assumed spectral (and angular) shape of the solution
must be used. In practice, these quantities are pre-calculated by nuclear data
processing software such as NJOY :cite:`njoy` or Fudge :cite:`fudge` and are
assumed to be given. For this work, the neutron velocity :math:`v_g^{-1}` is
calculated from the relativistic energy formula in Equation :eq:`velocity-04` at
the midpoint of the energy bin and Fudge has been used to generate all "full
model" neutron cross sections and energy grids.

Taking both the multigroup and discrete ordinate methods, the semi-discrete
neutron transport equation discretizes Equation :eq:`nte-continuous` for both
energy and angle. If the angular flux function is defined as

.. math::

   \psi_{m,g}(x,t) \approx \int_{E_{g-1}}^{E_{g}} dE \, \Psi(\bx, \bsOmega_m, E, t),\qquad g\in \cG:=\{1 ,\dots,G\},

it is the solution of the neutron transport equation

.. math::
   :label: nte-discrete

   \frac{1}{v_g} \frac{\partial}{\partial t}{\psi}_{m, g} + \bsOmega \cdot \nabla \psi_{m, g} + \sig[g]{t} \psi_{m, g} = \sum_{g'=1}^{G} \sig[g' \rightarrow g]{s} \phi_{g'} + \chi_{g} \sum_{g'=1}^{G} \nu_{g'} \sig[g' \rightarrow g]{f} \phi_{g'} + q_{m,g},

for angle :math:`m` and energy group :math:`g`. A quadrature rule is used to
relate the angular flux to the scalar flux and shown as

.. math::
   :label: nte-scalar

   \phi_g = \sum_{m=1}^{M_{\text{Angles}}} w_{m} \psi_{m,g},

where :math:`w` is the quadrature weights summing to one. The initial condition
for :math:`\psi_{m, g}` is

.. math::

   \psi_{m, g} (\bx,0) = f_{m, g} (\bx) \hquad \text{for} \hquad \bx \in D, \hquad m \in \cM, \hquad g\in\cG,

and the incoming boundary data is

.. math::

   \psi_{m, g}(\bx,t) = b_{m, g} ( \bx, t)  \hquad \text{for} \hquad \bn(\bx) \cdot \bsOmega_m < 0, \hquad m \in \cM, \hquad g\in\cG, \hquad t > 0,

where

.. math::

   f_{m, g}(\bx) = \int_{E_{g-1}}^{E_g} dE \, f(\bx,\bsOmega_m,E)

and

.. math::

   b_{m, g} = ( \bx, t)\int_{E_{g-1}}^{E_g} dE \, b(\bx,\bsOmega_m,E,t).

Equation :eq:`nte-discrete` is a set of :math:`M \cdot G` PDEs that will be
further discretized in both space and time in
:ref:`the spatial <sec-nte-spatial>` and
:ref:`temporal <sec-nte-temporal>` discretization sections.


Fixed Source Neutron Transport Equation
---------------------------------------

The multigroup discrete ordinates neutron transport equation presented in
Equation :eq:`nte-discrete` is the time-dependent neutron transport equation. The
time dependent term can be dropped from the equation to create a fixed source or
steady-state neutron transport equation. In both steady-state and time dependent
instances, the neutron flux is solved in the same manner. The neutron transport
is broken down into a system of linear equations for each energy group :math:`g`
and angle :math:`m` where there are two levels of iteration. The inner iteration
updates the scalar flux inside each energy group using a fixed-point source
iteration scheme, where the in-group scattering source is lagged and each update
requires the inversion of the operator that models streaming and loss. These
updates, called transport sweeps, amount to the inversion of a block triangular
system in space for each angle :cite:`o1987transport`. The outer iterations over
the energy variable often follow a Gauss-Seidel strategy that updates the
scattering and fission source terms as the energy groups are updated in order
from highest to lowest energies :cite:`bell1970nuclear`. The Gauss-Seidel method
is effective in problems that are dominated by down scattering, or lower
triangular scattering matrices. It is however, dependent on the number of
interactions that the neutrons have with the material, such as scattering and, in
time-dependent problems, fission. For example, a purely down-scattering material
without fission will converge in one Gauss-Seidel iteration. If there is
up-scattering or fission, the number of iterations can become prohibitively large
in optically thick problems :cite:`lewis1993`. The fixed source neutron transport
solver is explained in :ref:`the source-iteration algorithm <alg-source-iteration>`
where the convergence tolerances are :math:`\varepsilon_{G} = 10^{-8}` and
:math:`\varepsilon_{M} = 10^{-12}`.

.. _alg-source-iteration:

.. admonition:: Algorithm — Steady-State Source Iteration of the Multigroup, Discrete Ordinates Equation

   .. math::

      \begin{array}{l}
      \textbf{Require: material properties } (\sig[g]{t}, \sig[g'\to g]{s}, q_g), \text{ initial guess } \Tilde{\phi}, \\
      \quad \text{discretization parameters } (\bsOmega_m, w_m), \text{ tolerances } \varepsilon_G, \varepsilon_M \\[2pt]
      \Delta_G \gets 1 + \varepsilon_G, \quad j \gets 0 \\
      \phi_g{}^{0} \gets \Tilde{\phi}_g \\
      \textbf{while } \Delta_G > \varepsilon_G \textbf{ do} \qquad \triangleright\ \text{Outer iteration } (j) \\
      \quad \textbf{for } g = 1,\dots,G \textbf{ do} \qquad \triangleright\ \text{Loop over groups} \\
      \quad\quad \Tilde{Q}_g \gets q_g + \displaystyle\sum_{g'=1}^{g-1} \sig[g'\to g]{s} \phi_{g'}{}^{j+1} + \sum_{g'=g+1}^{G} \sig[g'\to g]{s} \phi_{g'}{}^{j} \qquad \triangleright\ \text{Off-scattering} \\
      \quad\quad \Delta_M \gets 1 + \varepsilon_M, \quad \ell \gets 0 \\
      \quad\quad \phi_g{}^{j+1,0} \gets \phi_g{}^{j} \\
      \quad\quad \textbf{while } \Delta_M > \varepsilon_M \textbf{ do} \qquad \triangleright\ \text{Source iteration } (\ell) \\
      \quad\quad\quad \textbf{for } m = 1,\dots,M \textbf{ do} \qquad \triangleright\ \text{Loop over angles} \\
      \quad\quad\quad\quad \Tilde{Q}_{m,g} \gets \Tilde{Q}_g + \sig[g\to g]{s} \phi_g{}^{j+1,\ell} \qquad \triangleright\ \text{Self-scattering} \\
      \quad\quad\quad\quad \psi_{m,g}{}^{j+1,\ell+1} \gets \left(\bsOmega_m \cdot \nabla + \sig[g]{t}\right)^{-1} \Tilde{Q}_{m,g} \qquad \triangleright\ \text{Transport sweep} \\
      \quad\quad\quad \textbf{end for} \\
      \quad\quad\quad \phi_g{}^{j+1,\ell+1} \gets \displaystyle\sum_{m=1}^{M} w_m \psi_{m,g}{}^{j+1,\ell+1} \\
      \quad\quad\quad \Delta_M \gets \left\| \dfrac{\phi_g{}^{j+1,\ell+1} - \phi_g{}^{j+1,\ell}}{\phi_g{}^{j+1,\ell+1}} \right\|_2 \\
      \quad\quad\quad \ell \gets \ell + 1 \\
      \quad\quad \textbf{end while} \\
      \quad\quad \psi_{m,g}{}^{j+1} \gets \psi_{m,g}{}^{j+1,\ell}, \quad \phi_g{}^{j+1} \gets \phi_g{}^{j+1,\ell} \\
      \quad \textbf{end for} \\
      \quad \Delta_G \gets \left\| \dfrac{\phi{}^{j+1} - \phi{}^{j}}{\phi{}^{j+1}} \right\|_2 \qquad \triangleright\ \phi{}^{j+1} = [\phi_1{}^{j+1},\dots,\phi_G{}^{j+1}] \\
      \quad j \gets j + 1 \\
      \textbf{end while} \\
      \textbf{return } \phi \gets \phi_g{}^{j}
      \end{array}


:math:`k`-Eigenvalue Neutron Transport Equation
-----------------------------------------------

Another variation of the neutron transport equation is the :math:`k`-eigenvalue
criticality transport equation :cite:`lewis1993`. This type of problem focuses on
the neutron population from one generation to the next. Taking Equation
:eq:`nte-discrete`, the time-dependent and external source terms are dropped and a
:math:`k`-effective multiplication factor :math:`\keff` is added to the fission
source term. The resulting equation becomes

.. math::
   :label: nte-critical

   \bsOmega \cdot \nabla \psi_{m, g} + \sig[g]{t} \psi_{m, g} = \sum_{g'=1}^{G} \sig[g' \rightarrow g]{s} \phi_{g'} + \frac{\chi_{g}}{\keff} \sum_{g'=1}^{G} \nu_{g'} \sig[g' \rightarrow g]{f} \phi_{g'},

where the boundary conditions are

.. math::

   \psi (\bx, \bsOmega, E, t) = 0.

This eigenvalue problem estimates the trend of the neutron population where
:math:`\keff = 1` represents no neutron population growth (critical system) while
:math:`\keff > 1` and :math:`\keff < 1` represent growth (supercritical) and
decay (subcritical), respectively. The criticality problems use Gauss-Seidel for
the inner loop but requires an additional step. This step uses the power
iteration method to calculate the largest :math:`k`-eigenvalue. In this approach,
the scattering term was updated with Gauss-Seidel while the fission term remained
fixed. After converging the scattering term, the fission term is updated with the
power iteration.

The power iteration assumes that the eigenvalue problem has a non-negative and
unique solution, :math:`\keff > 0` :cite:`lewis1993`. The outer iteration
requires an initial guess for the scalar flux :math:`\phi` and its associated
:math:`k`-effective multiplication factor :math:`\keff`. For the typical
:math:`k`-eigenvalue problem, a random matrix is used to initialize the scalar
flux. While the convergence tolerance has not been met, the fission source is
calculated from the scalar flux and :math:`\keff` value and sent to the inner
source iteration. The source iteration iterates over all energy groups :math:`G`
going from lowest energy level to highest energy level. For each energy group, a
source term is used to iterate over the discrete ordinates angles :math:`M` and
is updated using the Gauss-Seidel method. Once this scalar flux is sufficiently
converged for all angles and energy groups, it is returned to the power iteration
to update the scalar flux and :math:`\keff` terms and checking for outer
convergence using the :math:`\ell_{2}` norm. This process is fully explained in
:ref:`the power-iteration algorithm <alg-power-iteration>`. For these problems,
the outer convergence tolerance is set at :math:`\varepsilon_{K} = 10^{-6}`.
While there are different and faster methods to solve for the :math:`k`-eigenvalue
transport problem, these are some of the most widely used
techniques :cite:`lewis1993`.

.. _alg-power-iteration:

.. admonition:: Algorithm — :math:`k`-Eigenvalue Power Iteration of the Multigroup Discrete Ordinates Equation

   .. math::

      \begin{array}{l}
      \textbf{Require: material properties } (\sig[g]{t}, \sig[g'\to g]{s}, \chi_g, \nu_{g'}, \sig[g'\to g]{f}), \\
      \quad \text{initial guesses } (\Tilde{\phi}, \Tilde{\keff}), \text{ discretization parameters } (\bsOmega_m, w_m), \text{ tolerance } \varepsilon_K \\[2pt]
      \Delta_K \gets 1 + \varepsilon_K, \quad k \gets 1 \\
      \phi{}^{(0)} \gets \Tilde{\phi}, \quad \keff{}^{(0)} \gets \Tilde{\keff} \\
      \textbf{while } \Delta_K > \varepsilon_K \textbf{ do} \\
      \quad q = \dfrac{1}{\keff{}^{(k-1)}} \chi_g \displaystyle\sum_{g'=1}^{G} \nu_{g'} \sig[g'\to g]{f} \phi_{g'}{}^{(k-1)} \\
      \quad \phi{}^{(k)} \gets \text{source-iteration algorithm with } q_g \gets q,\ \Tilde{\phi} \gets \phi{}^{(k-1)} \qquad \triangleright\ \text{Source iteration} \\
      \quad \keff{}^{(k)} = \dfrac{\keff{}^{(k-1)} \chi_g \sum_{g'=1}^{G} \nu_{g'} \sig[g'\to g]{f} \phi_{g'}{}^{(k)}}{\chi_g \sum_{g'=1}^{G} \nu_{g'} \sig[g'\to g]{f} \phi_{g'}{}^{(k-1)}} \qquad \triangleright\ \text{Update } \keff \\
      \quad \phi{}^{(k)} \gets \phi{}^{(k)} \left[\displaystyle\sum_{i=1}^{I} \sum_{g=1}^{G} \left(\phi{}^{(k)}\right)^2\right]^{-1/2} \qquad \triangleright\ \text{Normalize flux} \\
      \quad \Delta_K \gets \left\| \dfrac{\phi{}^{(k)} - \phi{}^{(k-1)}}{\phi{}^{(k)}} \right\|_2 \\
      \quad k \gets k + 1 \\
      \textbf{end while} \\
      \textbf{return } \phi \gets \phi_g{}^{(k)}, \quad \keff \gets \keff{}^{(k)}
      \end{array}
