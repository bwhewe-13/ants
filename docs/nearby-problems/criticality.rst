.. _sec-mnp-criticality:

The Method of Nearby Problems for k-Eigenvalue Problems
=======================================================

The method of nearby problems can be seamlessly applied to fixed source neutron
transport problems. This is due to the presence of the external source term in
:eq:`mnp-discrete` and the simple addition of the discrete residual. In
:math:`k`-eigenvalue problems, there is no external source term present given the
nature of the criticality problems :cite:`lewis1993`. The discrete residual must
be included in the criticality problem to solve for the nearby flux utilizing a
technique different from adding it to the fixed source problems. The solver used
for calculating the neutron flux for :math:`k`-eigenvalue problems must be
adjusted to allow for the incorporation of the residual source term.

The :math:`k`-eigenvalue neutron transport equation is introduced in the
:ref:`neutron transport chapter <chapter-neutron-transport>` with
:eq:`nte-critical`. Using the power iteration, the numerical angular flux and
numerical :math:`\keff` value are calculated. The curve fit flux
:math:`S_{m, g}` is calculated in the same manner as in the
:ref:`fixed-source formulation <sec-mnp-fixed-source>`. For the analytical curve
fit, the associated curve fit :math:`\keff^{*}` term can be calculated as

.. math::

   \keff^{*} = \left[ \sum_{i=1}^{I} \left(\chi_{g} \sum_{g'=1}^{G} \nu_{g'}
   \sig[g' \rightarrow g]{f} \overline{S}_{g'} \right) \right]
   \left[ \sum_{i=1}^{I} \left(\bsOmega \cdot \nabla S_{m, g}
   + \sig[g]{t} S_{m, g} -  \sum_{g'=1}^{G} \sig[g' \rightarrow g]{s}
   \overline{S}_{g'} \right) \right]^{-1},

which can be used for the :math:`\keff` error calculation. The fission source
term :math:`q_{g}^{*}`,

.. math::

   q_{g}^{*} = \frac{1}{\keff^{*}} \chi_{g} \sum_{g'=1}^{G} \nu_{g'}
   \sig[g' \rightarrow g]{f} \overline{S}_{g'},

is used to replace the fission and external source terms in :eq:`mnp-residual` to
calculate the continuous residual. The modified fission source term does not have
an angular dependence, meaning the related spatially continuous :math:`R_{g}` and
discrete :math:`r_{g}` residuals do not have angular dependence.

.. _alg-mnp-power-iteration:

.. admonition:: Algorithm — k-Eigenvalue Method of Nearby Problems Power Iteration of the Multigroup Discrete Ordinates Equation

   .. math::

      \begin{array}{l}
      \textbf{Require: material properties } (\sig[g]{t}, \sig[g'\to g]{s}, \chi_g, \nu_{g'}, \sig[g'\to g]{f}), \\
      \quad \text{initial guess } (\widetilde{\phi}), \text{ discretization parameters } (\bsOmega_m, w_m), \\
      \quad \text{curve fit parameters } (\overline{S}_g, \keff^{*}), \text{ residual } (r_g), \text{ tolerance } \varepsilon_K \\[2pt]
      \Delta_K \gets 1 + \varepsilon_K, \quad k \gets 1 \\
      \phi{}^{(0)} \gets \widetilde{\phi} \\
      \keff{}^{(1/2)} \gets \keff^{*} \left[ \displaystyle\sum_{i=1}^{I} \left(\chi_g \displaystyle\sum_{g'=1}^{G} \nu_{g'} \sig[g'\to g]{f} \phi_{g'}^{(0)} \right) \right] \left[ \displaystyle\sum_{i=1}^{I} \left(\chi_g \displaystyle\sum_{g'=1}^{G} \nu_{g'} \sig[g'\to g]{f} \overline{S}_{g'} \right) \right]^{-1} \\
      \textbf{while } \Delta_K > \varepsilon_K \textbf{ do} \\
      \quad q = \dfrac{1}{\keff{}^{(k-1)}} \chi_g \displaystyle\sum_{g'=1}^{G} \nu_{g'} \sig[g'\to g]{f} \phi_{g'}{}^{(k-1)} + r_g \\
      \quad \phi{}^{(k)} \gets \text{source-iteration algorithm with } q_g \gets q,\ \widetilde{\phi} \gets \phi{}^{(k-1)} \qquad \triangleright\ \text{Source iteration} \\
      \quad \keff{}^{(k)} = \dfrac{\keff{}^{(k-1)} \chi_g \sum_{g'=1}^{G} \nu_{g'} \sig[g'\to g]{f} \phi_{g'}{}^{(k)}}{\chi_g \sum_{g'=1}^{G} \nu_{g'} \sig[g'\to g]{f} \phi_{g'}{}^{(k-1)}} \qquad \triangleright\ \text{Update } \keff \\
      \quad \phi{}^{(k)} \gets \phi{}^{(k)} \left[\displaystyle\sum_{i=1}^{I} \sum_{g=1}^{G} \left(\phi{}^{(k)}\right)^2\right]^{-1/2} \qquad \triangleright\ \text{Normalize flux} \\
      \quad \Delta_K \gets \left\| \dfrac{\phi{}^{(k)} - \phi{}^{(k-1)}}{\phi{}^{(k)}} \right\|_2 \\
      \quad k \gets k + 1 \\
      \textbf{end while} \\
      \textbf{return } \phi \gets \phi_g{}^{(k)}, \quad \keff \gets \keff{}^{(k)}
      \end{array}

To incorporate the discretized residual :math:`r_{g}` into the criticality
calculation in :eq:`nte-critical`, the power iteration method must be altered.
The standard power iteration, explained in the :ref:`power-iteration algorithm
<alg-power-iteration>`, is changed to allow for the residual term to be added to
each iteration. This can be seen in the :ref:`MNP power-iteration algorithm
<alg-mnp-power-iteration>`, which utilizes the method developed by
:cite:`wang2018` including the residual at every outer iteration. Another
important difference is that the :math:`\keff` value is initialized with the
curve fit scalar flux :math:`\overline{S}_{g}` and :math:`\keff^{*}` values. This
is indicated by :math:`\keff{}^{(1/2)}` to show that it comes after the angular
flux initialization and is consistent with the nomenclature of :cite:`wang2018`.
Employing the modified power iteration with the residual and curve fit
parameters, the nearby angular flux :math:`\psi\mnp` and nearby :math:`k`-effective
value :math:`\keff\mnp` can be calculated. The spatial discretization error
between the nearby flux and the curve fit flux can be used with
:eq:`mnp-error-equation` to ensure that the numerical solution is sufficiently
discretized.

The modified power iteration needed to run the method of nearby problems leads to
a more involved approach when using this technique with the Monte Carlo MC/DC
solver :cite:`morgan2024`. This is not a topic that has been investigated in this
analysis, and the Monte Carlo solver is used with the fixed source problems. The
:math:`k`-eigenvalue problem for Monte Carlo based solvers is an area for future
work. When applying the method of nearby problems to criticality problems, only a
discrete ordinates solver is employed.
