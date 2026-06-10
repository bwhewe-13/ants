.. _chapter-hybrid-method:

The Collision-Based Hybrid Method
=================================================

Collision-based hybrid algorithms have been explored and implemented in
neutron :cite:`hauck2013` and radiative :cite:`heningburg2020` transport
problems. This hybrid method is pursued to improve the computational time while
keeping a similar discretization error to high fidelity grids. Separating this
equation into a fine, uncollided grid and a coarse, collided grid allows for
faster wall clock times while keeping the error lower than traditional methods.
The collision-based hybrid method has been implemented for the neutron transport
equation for mono-energetic and multigroup problems and expands on previous
research :cite:`hauck2013,whewell2023`.


Introduction
------------

The neutron transport equation (NTE) is used to model neutron populations
traveling through different media. A common computational technique used to
solve this equation numerically is the discrete ordinates (:math:`S_N`)
method :cite:`lewis1993,chandrasekhar2013`. In a multigroup discrete ordinates
calculation, time-dependent problems are often discretized implicitly. This
results in a steady-state problem at each time step in the computation that is
solved iteratively, which is presented in
:ref:`the source-iteration algorithm <alg-source-iteration>`. The cost of a
multigroup calculation is a function of the energy resolution needed; coarsening
the energy grid, thus reducing the number of energy groups, can lead to less
expensive solutions. A smaller energy group structure must be carefully chosen
to limit the discretization error and preserve the characteristics of the
energy-dependent cross section :cite:`njoy`. There can be faster convergence
times using a low number of angles, but this can also lead to both larger errors
and exacerbate the non-physical ray effects that can arise in :math:`S_N`
calculations. There are techniques for reducing these ray
effects :cite:`hauck2019filtered,frank2020`, but the most common solution is to
increase the number of angles :cite:`lewis1993`, thus increasing the
computational time.

Acceleration techniques, such as coarse mesh rebalancing and diffusion synthetic
acceleration (DSA), are used to reduce the computational cost of the iterative
solver. These methods cannot be implemented indiscriminately, as coarse mesh
rebalancing must be concerned with the coarse mesh size :cite:`lewis1993` while
diffusion synthetic acceleration has difficulty in highly heterogeneous
materials :cite:`southworth2021`. For the outer iterations, there are two-grid,
nonlinear diffusion acceleration, and Krylov subspace schemes that can improve
the convergence of problems with upscattering or
fission :cite:`adams1993two,anistratov2013multilevel,slaybaugh2018`. While these
methods demonstrate improvements with high upscattering materials, there is a
nominal improvement over Gauss-Seidel in low upscattering
materials :cite:`adams1993two`. These methods must also include the coarse grid
diffusion equation solver for their transport code bases, something that is not
needed for the collision-based hybrid method.

More recently, collision-based hybrid algorithms for time-dependent transport
equations have been important topics for transport
researchers :cite:`hauck2013,crockatt2017arbitrary,crockatt2019hybrid,crockatt2020improvements,heningburg2020,whewell2023`.
These algorithms split the transport equation into collided and uncollided
components, as was done much earlier in the steady-state
context :cite:`alcouffe1977`. In the aforementioned examples, the hybrid
approach yielded substantial improvements in efficiency when compared to
monolithic discretization schemes. While previous work has applied different
spatial discretization schemes to the collided and uncollided components, the
approach has so far only been implemented on one-dimensional problems. In this
analysis, the collision-based hybrid algorithm is extended to the
two-dimensional multigroup setting. The algorithm is first applied to
one-dimensional slabs and spheres with different boundary conditions and sources
before including two-dimensional examples using a Cartesian grid. The hybrid
algorithm is shown to be more accurate for the same computational time or reach
solutions with the same error level while using significantly less time. As
shown in :cite:`whewell2023`, the application of a collided-uncollided splitting
to a monolithic discretization grid recovers the same solution as standard
Gauss-Seidel in less time. This wall clock improvement is due to the changes in
the solver that reduce unnecessary iterations in the inner loop of the nested
approach, as explored in :cite:`senecal2017`.

This chapter is organized as follows. The continuous and discrete
collided-uncollided split and the collision-based hybrid algorithm are presented
in :ref:`the hybrid formulation <sec-hy-formulation>`, building on the
continuous and multigroup discrete ordinates transport equations found in
:doc:`../neutron-transport/index`. Numerical results for one-dimensional test
problems, initially presented in :cite:`whewell2023`, are given in
:doc:`one-dimensional` and compare the hybrid method to more traditional
approaches. Two-dimensional results and conclusions, which demonstrate the
abilities of the hybrid method in more computationally expensive examples, are
presented in later sections.


.. _sec-hy-formulation:

The Collision-Based Hybrid Formulation
--------------------------------------

The goal of the hybrid approach is to accelerate the computation of a numerical
solution for the neutron transport equation :eq:`nte-continuous` relative to a
monolithic discretization, while minimizing the subsequent loss in accuracy. The
formulation of the hybrid method is best understood at the continuous level as a
splitting method. Since it is linear, the neutron transport equation
:eq:`nte-continuous` can be separated into two equations where a distinct
angular flux is calculated in each and the results added together to obtain the
total angular flux. For the nomenclature, the collided portions of the neutron
transport equation are represented with a superscript :math:`\mathrm{c}` while
the uncollided portions are represented with a superscript :math:`\mathrm{u}`.


The Continuous Collided-Uncollided Split
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first equation governs the uncollided angular flux, :math:`\Psiu`. It takes
the form

.. math::

   \left( \frac{1}{v(E)}\frac{\partial}{\partial t} + \Omega \cdot \nabla + \sig{t}(\bx, E) \right) \Psiu(\bx, \bsOmega,E,t) = \Qu(\bx,\bsOmega, E, t),

with :math:`\Qu = Q`. The initial condition for :math:`\Psiu` is

.. math::

   \Psiu(\bx, \bsOmega, E, 0) = f( \bx, \bsOmega, E) \hquad \text{for} \hquad \bx \in D, \hquad \bsOmega \in \bbS^2, \hquad E>0,

and the incoming boundary data is

.. math::

   \Psiu(\bx, \bsOmega, E, t) = b( \bx, \bsOmega, E, t) \hquad
   \text{for} \hquad \bx \in \partial D, \hquad  \bn(\bx) \cdot \bsOmega < 0, \hquad E>0, \hquad t >0.

The second equation governs the collided angular flux, :math:`\Psic`. It takes
the form

.. math::

   \begin{aligned}
   \left( \frac{1}{v(E)}\frac{\partial}{\partial t} + \Omega \cdot \nabla + \sig{t}(\bx, E) \right) & \Psic(\bx,\bsOmega,E,t) = \\
   & \hspace{-50pt} \int_{0}^{\infty} dE' \; \sig{s} (\bx, E' \rightarrow E) \Phic(\bx, E', t) +  \Qc(\bx,\bsOmega, E, t) \\
   & \hspace{-50pt} + \chi(\bx, E) \int_{0}^{\infty} dE' \; \nu(x, E') \sig{f}(\bx, E')  \Phic (\bx, E', t),
   \end{aligned}

where the isotropic term :math:`\Qc` comes from the scattering and fission
source of the uncollided flux:

.. math::

   \begin{aligned}
   \Qc(\bx, E, t) &= \int_{0}^{\infty}dE' \; \sig{s} (\bx, E' \rightarrow E) \Phiu(\bx, E', t) \\
   &+ \chi(\bx, E) \int_{0}^{\infty} dE' \; \nu(\bx, E') \sig{f}(\bx, E')  \Phiu(\bx, E', t).
   \end{aligned}

The initial condition for :math:`\Psic` is

.. math::

   \Psic(\bx, \bsOmega, E, 0) = 0 \hquad \text{for} \hquad \bx \in D, \hquad \bsOmega \in \bbS^2, \hquad E>0,

and the incoming boundary data is

.. math::

   \Psic(\bx, \bsOmega, E, t) = 0 \hquad \text{for} \hquad \bx \in \partial D, \hquad \bn(\bx) \cdot \bsOmega < 0, \hquad E>0, \hquad t >0.

A third equation for the total angular flux :math:`\Psit` takes the form

.. math::
   :label: total-angular-flux

   \left( \frac{1}{v(E)}\frac{\partial}{\partial t} + \Omega \cdot \nabla + \sig{t} \right) \Psit(\bx,\bsOmega,E,t) = \Qt(\bx,\bsOmega, E, t),

where the total external source :math:`\Qt` is

.. math::

   \begin{aligned}
   \Qt(\bx,\bsOmega, E, t) &= \int_{0}^{\infty}dE' \; \sig{s} (\bx, E' \rightarrow E) \Phic(\bx, E', t) + \Qc(\bx,\bsOmega, E, t) \\
   &+ \chi(\bx, E) \int_{0}^{\infty} dE' \; \nu \sig{f}(\bx, E') \Phic(\bx, E', t).
   \end{aligned}

The initial condition for :math:`\Psit` is

.. math::

   \Psit(\bx, \bsOmega, E, 0) = f( \bx, \bsOmega, E), \hquad \text{for} \hquad \bx \in D, \hquad \bsOmega \in \bbS^2, \hquad E>0,

and the incoming boundary data is

.. math::

   \Psit(\bx, \bsOmega, E, t) = b( \bx, \bsOmega, E, t) \hquad \text{for} \hquad \bx \in \partial D, \hquad \bn(\bx) \cdot \bsOmega < 0, \hquad E>0, \hquad t >0.

The continuous neutron transport equation implies that
:math:`\Psit = \Psiu + \Psic = \Psi`, making an independent equation for
:math:`\Psit` trivially redundant. This is no longer the case when equations for
:math:`\Psit`, :math:`\Psiu`, and :math:`\Psic` are discretized with different
methods and/or grids. In practice, the hybrid method seeks at each time step a
numerical approximation for :math:`\Psiu` on a fine grid and a numerical
approximation for :math:`\Psic` on a coarse grid. In the original
formulation :cite:`hauck2013`, these two approximate solutions were, at the end
of the time step, combined via a remapping procedure in which an approximation
for :math:`\Psic` is reconstructed on the fine grid. Unfortunately, the
reconstruction can introduce artifacts, particularly for :math:`S_N`
discretizations in multi-dimensional settings. To address this issue, the idea
of using the equation for :math:`\Psit` to remap onto the fine grid was
introduced in :cite:`crockatt2020improvements` since the source :math:`\Qt`
depends only on integrated quantities that can be computed on both grids. An
alternative view is that the collided-uncollided split provides a cheap method
for approximating the scattering and fission sources that appear in the
right-hand side of the neutron transport equation :eq:`nte-continuous`.

The choices of initial conditions, boundary conditions, and sources for
:math:`\Psiu` and :math:`\Psic` are not unique. The strategy above is to assign
data to the uncollided equation since it is equipped with the finest
discretization. This strategy may not always be the best choice, particularly in
highly-collisional problems with strong boundary layers :cite:`Densmore2006cla`.

The hybrid method is summarized in
:ref:`the backward Euler algorithm <alg-hybrid-method>` below.

.. _alg-hybrid-method:

.. admonition:: Algorithm — Backward Euler Time-Dependent Collision-Based Hybrid Method (multigroup, discrete ordinates)

   .. math::

      \begin{array}{l}
      \textbf{Require: uncollided properties } (\sig[g]{t}, \sig[g'\to g]{s}, \chi_g, \nu_{g'}, \sig[g'\to g]{f}, v_g, q_g), \\
      \quad \textbf{collided properties } (\hsig[\hg]{t}, \hsig[\hg'\to\hg]{s}, \hat\chi_{\hg}, \hat\nu_{\hg'}, \hsig[\hg'\to\hg]{f}, v_{\hg}), \\
      \quad \text{previous step } \psi_{m,g}{}^{(n-1)}, \text{ parameters } (\Delta t, \bsOmega_m, w_m, \hOmega_{\hm}, \hw_{\hm}, \Delta E_g, \Delta\hE_{\hg}), \text{ tolerances } \varepsilon_G, \varepsilon_M \\[4pt]
      \psiu_{m,g}{}^{(n-1)} \gets \psi_{m,g}{}^{(n-1)}, \quad \qu_{g} \gets q_{g} \\[2pt]
      \textbf{for } g = 1,\dots,G \textbf{ do} \qquad \triangleright\ \text{Uncollided flux update} \\
      \quad \textbf{for } m = 1,\dots,M \textbf{ do} \\
      \quad\quad \psiu_{m,g} \gets \left(\dfrac{1}{v_g \Delta t} + \bsOmega_m \cdot \nabla + \sig[g]{t}\right)^{-1} \left(\qu_{g} + \dfrac{1}{v_g \Delta t} \psiu_{m,g}{}^{(n-1)}\right) \quad \triangleright\ \text{Transport sweep} \\
      \quad \textbf{end for} \\
      \textbf{end for} \\
      \phiu_g \gets \displaystyle\sum_{m=1}^{M} w_m \psiu_{m,g} \\[4pt]
      \qc_{\hg} \gets \displaystyle\sum_{g=\Gamma_{\hg}+1}^{\Gamma_{\hg+1}} \sum_{g'=1}^{G} \sig[g'\to g]{s} \phiu_{g'} + \sum_{g=\Gamma_{\hg}+1}^{\Gamma_{\hg+1}} \chi_g \sum_{g'=1}^{G} \nu_{g'} \sig[g'\to g]{f} \phiu_{g'} \quad \triangleright\ \text{Collided source} \\[4pt]
      \phic_{\hm,\hg} \gets \text{time-dependent algorithm with } \sig[g]{t}\gets\hsig[\hg]{t},\ \sig[g'\to g]{s}\gets\hsig[\hg'\to\hg]{s},\ \chi_g\gets\hat\chi_{\hg}, \\
      \quad \nu_{g'}\gets\hat\nu_{\hg'},\ \sig[g'\to g]{f}\gets\hsig[\hg'\to\hg]{f},\ v_g\gets v_{\hg},\ q_g\gets\qc_{\hg},\ \bsOmega_m\gets\hOmega_{\hm},\ w_m\gets\hw_{\hm},\ \psi^{(n-1)}\gets 0 \quad \triangleright\ \text{Collided flux update} \\[4pt]
      \qt_{m,g} \gets \dfrac{\Delta E_g}{\Delta\hE_{\hg}} \left[\qc_{\hg} + \displaystyle\sum_{\hg'=1}^{\hG} \sig[\hg'\to\hg]{s} \phic_{\hg'} + \chi_{\hg} \sum_{\hg'=1}^{\hG} \nu_{\hg'} \sig[\hg'\to\hg]{f} \phic_{\hg'}\right] \quad \triangleright\ \text{Total source} \\[4pt]
      \textbf{for } g = 1,\dots,G \textbf{ do} \qquad \triangleright\ \text{Total flux update} \\
      \quad \textbf{for } m = 1,\dots,M \textbf{ do} \\
      \quad\quad \psit_{m,g} \gets \left(\dfrac{1}{v_g \Delta t} + \bsOmega_m \cdot \nabla + \sig[g]{t}\right)^{-1} \qt_{m,g} \quad \triangleright\ \text{Transport sweep} \\
      \quad \textbf{end for} \\
      \textbf{end for} \\
      \psi_{m,g}{}^{(n)} \gets \psit_{m,g}, \quad \phi_g{}^{(n)} \gets \displaystyle\sum_{m=1}^{M} w_m \psit_{m,g} \\
      \textbf{return } \psi_{m,g}{}^{(n)},\ \phi_g{}^{(n)}
      \end{array}


The Hybrid Multigroup, Discrete Ordinates Equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the hybrid approach, the multigroup, discrete ordinates approximation is
applied to the uncollided, collided, and total flux equations. It can be assumed
that the uncollided and total flux equations are discretized at the same high
resolution using :math:`G` groups and :math:`M` angles, while the collided
equation uses lower resolution :math:`\hG` and :math:`\hM` angles. To
differentiate between different discretization parameters, a hat adornment
:math:`(\hat{\cdot})` is used for the collided parameters. The weights and angles
in the discretization of the collided equations are denoted by :math:`\hw_{\hm}`
and :math:`\hOmega_{\hm}`, respectively, for :math:`\hm = 1, \dots, \hM`. To
define the coarse groups, let

.. math::

   0 = \Gamma_0 < \Gamma_1 \dots < \Gamma_{\hat{g}} \dots < \Gamma_{\hG} = G

be a set of :math:`\hG + 1` integers and set :math:`\hE_{\hg} = E_{\Gamma_{\hat g}}`.
Then for each :math:`\hat{g} \in \chG`,

.. math::

   \Delta \hE_{\hg}
   = \hE_{\hg} - \hE_{\hg-1}
   = E_{\Gamma_{\hat g}} - E_{\Gamma_{\hat g-1}}
   = \sum_{g=\Gamma_{\hg-1}+1}^{ \Gamma_{\hg}}
   E_{g} - E_{g-1}
   = \sum_{g=\Gamma_{\hg-1}+1}^{ \Gamma_{\hg}} \Delta E_g.

The uncollided equation for

.. math::

   \psiu_{m,g} \approx \int_{E_{g-1}}^{E_{g}} dE \, \Psiu(\bx, \bsOmega_m, E, t)

is

.. math::
   :label: hy-uncollided

   \frac{1}{v_g} \frac{\partial }{\partial t} \psiu_{m, g} + \bsOmega_m \cdot \nabla \psiu_{m, g} + \sig[g]{t} \psiu_{m, g} = \qu_{m,g},

where :math:`\qu_{m,g} = q_{m,g}`. Equation :eq:`hy-uncollided` is solved over a
time-step :math:`[t^n, t^{n+1})` with initial condition

.. math::

   \psiu_{m, g} (\bx,t^n) =
   \begin{cases}
   f_{m, g}(\bx), & \bx \in D, \hquad m \in \cM, \hquad g\in\cG, \hquad t^n = 0 \\
   \psit_{m, g}(\bx,t^n_-), &  \bx \in D,\hquad m \in \cM, \hquad g\in\cG, \hquad t^n > 0
   \end{cases},

and incoming boundary data

.. math::

   \psiu_{m, g} = b_{m, g} (\bx, t) \hquad \text{for} \hquad \bn(\bx) \cdot \bsOmega_m < 0, \hquad m \in \cM, \hquad g \in \cG \quand t>0.

The collided transport equation for

.. math::

   \psic_{\hat m,\hat g} \approx \int_{E_{\hat g-1}}^{E_{\hat{g}}} dE \, \Psic(\bx, \hOmega_m, E, t)

is

.. math::
   :label: hy-collided

   \frac{1}{v_{\hg}} \frac{\partial }{\partial t} \psic_{\hm, \hg} + \hOmega_\hm \cdot \nabla \psic_{\hm, \hg} + \hsig[g]{t} \psic_{\hm, \hg} =
   \sum_{g'=1}^{\hG} \hsig[\hg' \rightarrow \hg]{s} \phic_{\hg'}
   + \hat \chi_{\hg} \sum_{\hg'=1}^{\hG} \hat \nu_{\hg'} \hsig[\hg' \rightarrow \hg]{f} \phic_{\hg'} + \qc_{\hg},

where

.. math::
   :label: hy-gg-coarsen-01

   \begin{aligned}
   v_{\hg} &= \frac{1}{\Delta \hat{E}_{\hg}} \sum_{g=\Gamma_{\hg-1}}^{\Gamma_{\hg}} v_g, \\
   \phic_{\hg'} &= \sum_{\hm=1}^{\hM}  \hw_{\hm} \psic_{\hm,\hg'}, \\
   \hat \chi_{\hg} &= \sum_{g=\Gamma_{\hg}+1}^{\Gamma_{\hg+1}} \chi_{g} \quand \\
   \hat \nu_{\hg'} &= \sum_{g=\Gamma_{\hg}+1}^{\Gamma_{\hg+1}} \nu_{g};
   \end{aligned}

the energy-coarsened cross-sections are given by

.. math::
   :label: hy-gg-coarsen-02

   \begin{aligned}
   \hsig[\hg]{t} &= \frac{1}{\Delta \hat{E}_{\hg}} \sum_{g=\Gamma_{\hg-1}+1}^{\Gamma_{\hg}} \Delta E_{g} \sig[g]{t} \quand \\
   \hsig[\hg]{\ell} &= \frac{1}{\Delta \hat{E}_{\hg}} \sum_{g=\Gamma_{\hg-1}+1}^{\Gamma_{\hg}} \sum_{g'=\Gamma_{\hg-1}+1}^{\Gamma_{\hg}} \Delta E_{g'} \sig[g'\rightarrow g]{\ell}, \quad \ell \in \{\mathrm{s},\mathrm{f}\};
   \end{aligned}

and the isotropic source is

.. math::
   :label: hy-q-collided

   \qc_{\hg} = \sum_{g=\Gamma_{\hg-1}+1}^{\Gamma_{\hg}}  \sum_{g'=1}^{G} \sig[g' \rightarrow g]{s} \phiu_{g'} + \sum_{g=\Gamma_{\hg-1}+1}^{\Gamma_{\hg}} \chi_{g} \sum_{g'=1}^{G} \nu_{g'} \sig[g' \rightarrow g]{f} \phiu_{g'},

with

.. math::

   \begin{aligned}
   \phiu_{g} &= \sum_{m=1}^M w_m \psiu_{m,g} \quand \\
   \phic_{\hat g} &= \sum_{\hat m=1}^{\hat M} \hw_{\hm} \psic_{\hat{m},\hat g}.
   \end{aligned}

We solve Equation :eq:`hy-collided` over a time-step :math:`[t^n, t^{n+1})` with
initial condition

.. math::

   \psic_{\hm, \hg} (\bx,t^n) = 0 \hquad \text{for} \hquad \bx \in D, \hquad \hm \in \chM:=\{1, \cdots, \hM \}, \hquad \hg \in \chG, \hquad t^n \geq 0

and boundary data

.. math::

   \psic_{\hm, \hg} = 0 \qquad \text{for} \quad \bn(\bx) \cdot \hOmega_{\hm} < 0, \quad \hm \in \chM, \quad \hg\in\chG, \quad t>0.

The total flux equation for

.. math::

   \psit_{m,g}(x,t) \approx \int_{E_{g-1}}^{E_{g}} dE \, \Psit(\bx, \bsOmega_m, E, t)

is

.. math::
   :label: hy-phi-total

   \frac{1}{v_g} \frac{\partial }{\partial t} \psit_{m, g} + \bsOmega_m \cdot \nabla \psit_{m, \hat g} + \sig[g]{t} \psit_{m, g} = \qt_{m,g}

where

.. math::

   \qt_{m,g} = \frac{\Delta E_g}{\Delta \hE_{\hg}} \left[\qc_{\hg} +
   \sum_{\hg'=1}^{\hG} \sig[\hg' \rightarrow \hg]{s} \phic_{\hg'} + \chi_{\hg} \sum_{\hg'=1}^{\hG} \nu_{\hg'} \sig[\hg' \rightarrow \hg]{f} \phic_{\hg'} \right]

and :math:`\hg` is the unique integer such that
:math:`\Gamma_{\hg-1} + 1 \leq g \leq \Gamma_{\hg}` or, equivalently,
:math:`(E_{g-1},E_{g}) \subset (\hE_{\hg-1},\hE_{\hg})`. We solve Equation
:eq:`hy-phi-total` over a time-step :math:`[t^n, t^{n+1})` with initial data

.. math::

   \psit_{m, g} (\bx,t^n) =
   \begin{cases}
   f_{m, g}(\bx), & \bx \in D, \hquad m \in \cM, \hquad g\in\cG, \hquad t^n = 0 \\
   \psit_{m, g}(\bx,t^n_-), & \bx \in D, \hquad m \in \cM, \hquad g\in\cG, \hquad t^n > 0
   \end{cases},

and incoming boundary condition

.. math::

   \psit_{m, g} = b_{m, g}(\bx, t) \qquad \text{for} \quad \bn(\bx) \cdot \bsOmega_m < 0, \quad m \in \cM, \quad g\in\cG\quand t>0.

In summary, Equations :eq:`hy-uncollided`, :eq:`hy-collided`, and
:eq:`hy-phi-total` are solved in succession for each time step. The solution of
Equation :eq:`hy-phi-total` at the end of the time step provides the initial
condition needed by Equations :eq:`hy-uncollided` and :eq:`hy-phi-total` at the
next time step to consistently join the uncollided and collided calculations.
The initial condition for Equation :eq:`hy-collided` is set to zero for each time
step.

The solution procedure for solving the collided flux is similar to the approach
used for solving the multigroup discrete ordinates neutron transport equation as
shown in :ref:`the time-dependent algorithm <alg-time-dependent>`. The hybrid
iteration for a backward Euler temporal discretization scheme is shown in
:ref:`the backward Euler algorithm <alg-hybrid-method>`. It relies spatially on
the very same sweeps and fixed-point source iteration for solving the flux at
each energy group, while Gauss-Seidel is used to integrate over
groups :cite:`bell1970nuclear,lewis1993`. The uncollided and the total flux
updates do not require any iterations because the right-hand side of the
respective equation is fixed.


Metrics for the Performance of the Hybrid Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The purpose of the collision-based hybrid method is to reduce the required
amount of computational time by coarsening the collided energy and angular
grids. The separation of the collided and uncollided portions of the neutron
transport equation is performed to increase the accuracy of these low fidelity
models as compared to a standard coarsening procedure. To show both the wall
clock speed up and the accuracy of the hybrid method over standard techniques,
three solution strategies are used for each test problem.

The first is referred to as the "Multigroup" solution, which is the traditional
neutron multigroup method with :math:`G` groups and :math:`M` angles. It uses
Gauss-Seidel with a tolerance of :math:`\varepsilon_G = 1 \times 10^{-8}` for the
outer iteration and source iteration with a tolerance of
:math:`\varepsilon_M = 1 \times 10^{-12}` for the inner iteration, as described
in :ref:`the source-iteration algorithm <alg-source-iteration>`. The coarsening
strategy in Equations :eq:`hy-gg-coarsen-01` and :eq:`hy-gg-coarsen-02` is used
for lower fidelity multigroup models. The accuracy of the multigroup method is
not necessarily monotonic in the number of groups. In particular, it can happen
that smaller number of groups yield better answers due to the nonlinearity of
the procedure, i.e., if the solution is separable in energy, space, and angle, a
single group calculation can be exact :cite:`bell1970nuclear`. There can be
cancellation of errors in integrated quantities, with these behaviors observed
in some test cases.

The second solver is the "Hybrid" method as described in
:ref:`this section <sec-hy-formulation>`. It uses :math:`G` groups and :math:`M`
angles in the discretization of the total and uncollided angular fluxes, and it
uses :math:`\hG` groups and :math:`\hM` angles in the discretization of the
collided angular flux. The hybrid method employs the algorithm described in
:ref:`the backward Euler algorithm <alg-hybrid-method>` with iteration
tolerances of :math:`\varepsilon_G = 1 \times 10^{-8}` and
:math:`\varepsilon_M = 1 \times 10^{-12}` for the outer and inner iterations,
respectively.

The final solver used is the "Splitting" method, which is the same as the hybrid
method described in :ref:`the backward Euler algorithm <alg-hybrid-method>`. The
same iteration tolerances (:math:`\varepsilon_G = 1 \times 10^{-8}` and
:math:`\varepsilon_M = 1 \times 10^{-12}`) are used but the energy groups and
angles remain consistent for the collided, uncollided, and total flux
(:math:`G = \hG` and :math:`M = \hM`). The splitting method demonstrates the
over-solving explored in :cite:`senecal2017` and exemplified
in :cite:`whewell2023`. The accuracy of the splitting method is equivalent to
the multigroup method and the accuracy will not be included in either the one- or
two-dimensional results. This method is included in the results to show the
inherent speed up of using the hybrid method over the multigroup method.

To compare the accuracy between the different models, various metrics are used.
The first is the total fission rate density (FRD) as described by

.. math::
   :label: fission-rate-density

   {\rm{FRD}} = \; \sum_{g=1}^{G} \left (\chi_{g} \sum_{g'=1}^{G} \nu_{g'} \sig[g' \rightarrow g]{f} \phi_{g'} \right),

which is the sum of the fission rate at each spatial error with units of
:math:`\mathrm{cm}^{-3}\,\mathrm{s}^{-1}`. The scattering rate density is also
used, in which the fission terms of Equation :eq:`fission-rate-density` are
replaced with the scattering matrix :math:`\sig{s}`. In certain instances, for
the two-dimensional results, the scalar flux is summed over all the energy
groups to account for regions that have neither scattering or fission. To
condense the accuracy into one term, the root mean squared error (RMSE) is used
with the fission rate density, scatter rate density, or scalar flux. There are
also cases where the energy groups are broken down into thermal, epithermal, and
fast energy regions. The thermal region includes all energy groups below 1 keV,
the epithermal region is between 1 keV and 1 MeV, and the fast region includes
all energy groups above 1 MeV. The error difference subtracts the hybrid error
from the multigroup error, meaning that for positive values, the hybrid method is
more accurate.

To compare the wall clock times between the different models, the formula is

.. math::
   :label: wall-clock-diff

   \tau' = \frac{\tau_{\rm{mg}} - \tau_{\rm{hy}}}{\tau_{\rm{mg}}},

where :math:`\tau_{\rm{mg}}` and :math:`\tau_{\rm{hy}}` are the wall clock times
required to run the multigroup and hybrid simulations, respectively. This format
was used to show that positive wall clock time differences are for faster hybrid
method simulation times. To calculate the wall clock time effectively, these
times were recorded five separate times and averaged. The one-dimensional wall
clock results used a personal computer while the two-dimensional results used the
high processing computers at Lawrence Livermore National Laboratory.

To combine both the wall clock time and accuracy into a single metric, a modified
figure of merit (FOM) is used as

.. math::
   :label: figure-of-merit

   \FOM = \frac{1}{\varepsilon \, \tau},

where :math:`\varepsilon` is the RMSE and :math:`\tau` is the wall clock time.
This is a metric commonly used in the Monte Carlo
community :cite:`figureofmerit` where :math:`\varepsilon^2` is used in place of
:math:`\varepsilon` in Equation :eq:`figure-of-merit` as it is the variance in a
statistical estimate. Since a larger figure of merit indicates a more efficient
calculation, the FOM difference subtracts the multigroup FOM from the hybrid FOM.
This is formulated to show that, for positive FOM difference, the hybrid method
is more desirable than the multigroup method.
