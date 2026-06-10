.. _sec-nte-continuous:

The Continuous Neutron Transport Equation
=========================================

The combination of terms describing the creation and destruction of neutrons
results in the neutron transport equation. In effect, the neutron transport
equation (NTE) models the behavior of advecting neutrons as they interact with
the surrounding medium :cite:`lewis1993`. The NTE combines the time-dependent,
collision, and streaming terms with the scattering, fission, and external
sources, with the :math:`d\bx`, :math:`d\bsOmega`, and :math:`dE` terms being
divided out. Rearranging, this results in the form :cite:`lewis1993`

.. math::
   :label: nte-continuous

   \begin{aligned}
   \left( \frac{1}{v(E)}\frac{\partial}{\partial t} + \bsOmega \cdot \nabla + \sig{t}(\bx, E) \right) & \Psi(\bx,\bsOmega,E,t) = \\
   & \hspace{-50pt} \int_{0}^{\infty}dE' \; \sig{s} (\bx, E' \rightarrow E) \Phi(\bx, E', t) +  Q(\bx,\bsOmega, E, t) \\
   & \hspace{-50pt} + \chi(\bx, E) \int_{0}^{\infty} dE' \; \nu(x, E') \sig{f}(\bx, E') \Phi (\bx, E', t),
   \end{aligned}

with the scalar flux represented as :math:`\Phi (\bx, E, t)` and obtained by
integrating the angular flux over angle. The angular flux depends on a spatial
coordinate :math:`\bx \in D \subset \bbR^3`, angular coordinate
:math:`\bsOmega \in \bbS^2`, energy :math:`E > 0`, and time :math:`t > 0`.

The initial condition for Equation :eq:`nte-continuous` is

.. math::

   \Psi(\bx, \bsOmega, E, t=0) = f( \bx, \bsOmega, E) \quad \text{for} \quad \bx \in D, \quad \bsOmega \in \bbS^2, \quad E > 0,

where :math:`f` is given. The incoming boundary data is prescribed as

.. math::

   \Psi(\bx, \bsOmega, E, t) = b( \bx, \bsOmega, E, t) \quad \text{for} \quad \bx \in \partial D, \quad  \bn(\bx) \cdot \bsOmega < 0, \quad E>0, \quad t >0,

where :math:`\bn (\bx)` is the unit outward normal at :math:`\bx \in \partial D`,
the boundary of :math:`D`.
