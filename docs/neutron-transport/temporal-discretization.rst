.. _sec-nte-temporal:

Temporal Discretization Schemes
===============================

Two different temporal discretization schemes are used with the neutron transport
equation in this work. The first is the backward Euler method, which is a simple,
first order method. The second is the Trapezoidal Rule with Second Order Backward
Difference (TR-BDF2), which is a second order method that combines both the
Crank-Nicolson (CN) and Second Order Backward Difference (BDF2) method. To explain
the TR-BDF2 scheme fully, both the CN and BDF2 schemes will be discussed.

To keep the notation simple, the streaming operator will not be expanded and the
:math:`g` subscripts will be dropped. In addition, the source terms on the
right-hand side of Equation :eq:`nte-discrete` will be combined into a single term
:math:`Q` resulting in

.. math::

   \frac{1}{v} \frac{\partial \psi}{\partial t} + \bsOmega \cdot \nabla \psi + \sig{t} \psi = Q \equiv \sig{s} \phi + \chi \nu \sig{f} \phi + q

as the time-dependent neutron transport equation. The non-time dependent terms
can be consolidated into a single operator as

.. math::
   :label: nte-temporal

   \frac{1}{v} \frac{\partial \psi}{\partial t} = \bA(\psi) \equiv Q - \bsOmega \cdot \nabla \psi - \sig{t} \psi,

which will be used with the various discretization schemes. The methods used for
the temporal discretization can be easily added to the spatial discretization
method discussed in :ref:`the spatial discretization section <sec-nte-spatial>`.
The fixed source source iteration scheme in
:ref:`the source-iteration algorithm <alg-source-iteration>` is modified for the
time dependent case and shown in
:ref:`the time-dependent algorithm <alg-time-dependent>`. For simplicity, the
backward Euler discretization scheme is shown.

.. _alg-time-dependent:

.. admonition:: Algorithm — Backward Euler Time Dependent Source Iteration of the Multigroup, Discrete Ordinates Equation

   .. math::

      \begin{array}{l}
      \textbf{Require: material properties } (\sig[g]{t}, \sig[g'\to g]{s}, \chi_g, \nu_{g'}, \sig[g'\to g]{f}, v_g, q_g), \\
      \quad \text{previous time step } \psi_{m,g}{}^{(n-1)}, \text{ parameters } (\Delta t, \bsOmega_m, w_m), \text{ tolerances } \varepsilon_G, \varepsilon_M \\[2pt]
      \Delta_G \gets 1 + \varepsilon_G, \quad j \gets 0 \\
      \phi_g{}^{0} \gets \displaystyle\sum_{m=1}^{M} w_m \psi_{m,g}{}^{(n-1)} \\
      \textbf{while } \Delta_G > \varepsilon_G \textbf{ do} \qquad \triangleright\ \text{Outer iteration } (j) \\
      \quad \textbf{for } g = 1,\dots,G \textbf{ do} \qquad \triangleright\ \text{Loop over groups} \\
      \quad\quad \Tilde{Q}_g \gets q_g + \displaystyle\sum_{g'=1}^{g-1} \sig[g'\to g]{s} \phi_{g'}{}^{j+1} + \sum_{g'=g+1}^{G} \sig[g'\to g]{s} \phi_{g'}{}^{j} \\
      \quad\quad \Tilde{Q}_g \gets \Tilde{Q}_g + \chi_g \displaystyle\sum_{g'=1}^{g-1} \nu_{g'} \sig[g'\to g]{f} \phi_{g'}{}^{j+1} + \chi_g \sum_{g'=g+1}^{G} \nu_{g'} \sig[g'\to g]{f} \phi_{g'}{}^{j} \\
      \quad\quad \Delta_M \gets 1 + \varepsilon_M, \quad \ell \gets 0 \\
      \quad\quad \phi_g{}^{j+1,0} \gets \phi_g{}^{j} \\
      \quad\quad \textbf{while } \Delta_M > \varepsilon_M \textbf{ do} \qquad \triangleright\ \text{Source iteration } (\ell) \\
      \quad\quad\quad \textbf{for } m = 1,\dots,M \textbf{ do} \qquad \triangleright\ \text{Loop over angles} \\
      \quad\quad\quad\quad \Tilde{Q}_{m,g} \gets \Tilde{Q}_g + \sig[g\to g]{s} \phi_g{}^{j+1,\ell} + \chi_g \nu_g \sig[g\to g]{f} \phi_g{}^{j+1,\ell} + \dfrac{1}{v_g \Delta t} \psi_{m,g}{}^{(n-1)} \\
      \quad\quad\quad\quad \psi_{m,g}{}^{j+1,\ell+1} \gets \left(\dfrac{1}{v_g \Delta t} + \bsOmega_m \cdot \nabla + \sig[g]{t}\right)^{-1} \Tilde{Q}_{m,g} \qquad \triangleright\ \text{Transport sweep} \\
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
      \psi_{m,g}{}^{(n)} \gets \psi_{m,g}{}^{j}, \quad \phi_g{}^{(n)} \gets \phi_g{}^{j} \\
      \textbf{return } \psi_{m,g}{}^{(n)}, \quad \phi_g{}^{(n)}
      \end{array}


Backward Euler
--------------

The backward Euler method is one of the more common discretization
schemes :cite:`butcher2016` in which the time derivative in Equation
:eq:`nte-temporal` is approximated as

.. math::

   \frac{1}{v} \frac{\partial \psi}{\partial t} = \frac{1}{v \, \Delta t} \left( \psi^{n+1} - \psi^{n} \right) = \bA(\psi^{n+1})

where the angular flux is known at time step :math:`n` and the time interval is
represented as :math:`\Delta t = t^{n+1} - t^{n}` when assuming a uniform temporal
grid. To relate this to the neutron transport equation in Equation
:eq:`nte-spatial`, the collision and time-dependent terms can be combined on the
left-hand side, while the external source and angular flux at the previous time
step can be combined on the right-hand side

.. math::

   \begin{aligned}
   \sig{*} \psi_{m}^{n+1} &= \left(\frac{1}{v \, \Delta t} + \sig{t}\right) \psi_{m}^{n+1} \\
   q_{m}^{*} &= q_{m}^{n+1} + \frac{1}{v \, \Delta t} \psi^{n}
   \end{aligned}

resulting in the first order accurate backward Euler discretization scheme.


Crank-Nicolson
--------------

The Crank-Nicolson (CN) temporal discretization scheme is unconditionally stable
and has been used extensively within the radiation transport
community :cite:`jovanovic2013,edwards2011`. This method is based off of the
trapezoidal rule, which takes an average of the angular flux at time steps
:math:`n` and :math:`n + 1` and assumes that the angular flux at time step
:math:`n` is known. For this method, the time-dependent term is approximated as

.. math::

   \frac{1}{v} \frac{\partial \psi}{\partial t} = \frac{1}{v \, \Delta t} \left( \psi^{n+1} - \psi^{n} \right) = \frac{1}{2} \left( \bA(\psi^{n+1}) + \bA(\psi^{n}) \right).

When fitting this to the form of Equation :eq:`nte-spatial`, the collision and
time-dependent terms can be combined on the left-hand side. The source term will
combine the source term from the previous time step :math:`n` and the current time
step :math:`n+1` as well as the balancing equation from the previous time step.
This results in

.. math::

   \begin{aligned}
   \sig{*} \psi_{m}^{n+1} &= \left(\frac{2}{v \Delta t} + \sig{t} \right) \psi_{m}^{n+1} \\
   q_{m}^{*} &= q_{m}^{n+1} + \frac{2}{v \Delta t} \psi^{n} + \sig{s} \, \phi^{n} + \chi \nu \sig{f} \, \phi^{n} + q_{m}^{n} - \bsOmega \cdot \nabla \psi^{n} - \sig{t} \psi^{n}_{m},
   \end{aligned}

which can be used with Equation :eq:`nte-spatial` for the second order accurate
Crank-Nicolson temporal discretization method.


Second Order Backward Differentiation Formula
---------------------------------------------

The second order Backward Differentiation Formula (BDF2) is a second order
accurate temporal discretization scheme that uses the known flux from the two
previous time steps :cite:`nishikawa2019`. When solving for time step
:math:`n + 1`, the flux at time steps :math:`n` and :math:`n - 1` must be known.
The BDF2 scheme is not self-starting and for the first time step :math:`n = 1`,
the backward Euler method is used before starting the BDF2 scheme. The BDF2 time
discretization is shown as

.. math::
   :label: nte-bdf2

   \frac{1}{v} \frac{\partial \psi}{\partial t} = \frac{1}{2 v \Delta t} \left(3 \psi^{n+1} - 4 \psi^{n} + \psi^{n-1} \right) = \bA(\psi^{n+1}),

where the time step :math:`n + 1` is solved for. Like the backward Euler scheme,
the collision and time-dependent terms can be combined on the left-hand side and
the external source and previous time step terms can be combined on the
right-hand side. This results in

.. math::

   \begin{aligned}
   \sig{*} \psi_{m}^{n+1} &= \left(\frac{3}{2 v \Delta t} + \sig{t}\right) \psi_{m}^{n+1} \\
   q_{m}^{*} &= q_{m}^{n+1} + \frac{2}{v \Delta t} \psi_{m}^{n} - \frac{1}{2 v \Delta t} \psi_{m}^{n-1}
   \end{aligned}

which can be combined into the source and collision terms of Equation
:eq:`nte-spatial`.


Trapezoidal Rule - Backward Differentiation Formula
---------------------------------------------------

There are certain pitfalls for both the Crank-Nicolson and BDF2 time
discretization schemes, such as oscillating results (CN) or increased error from
the backward Euler time step (BDF2), both of which can be observed with stiff
systems :cite:`edwards2011,nishikawa2019`. The Trapezoidal Rule with Second Order
Backward Difference (TR-BDF2) attempts to fix these issues by combining the CN and
BDF2 schemes into a single time step. If the flux is known at time step :math:`n`,
an intermediary time step :math:`n + \gamma` is taken using the Crank-Nicolson
scheme. This changes the governing equation in Equation :eq:`nte-temporal` into
the form

.. math::

   \frac{1}{v} \frac{\partial \psi}{\partial t} = \frac{1}{v \Delta t} \left( \psi^{n+\gamma} - \psi^{n} \right) = \frac{\gamma}{2} \left( \bA(\psi^{n+\gamma}) + \bA(\psi^{n}) \right),

where the angular flux is solved for time step :math:`n + \gamma`. As with the
Crank-Nicolson scheme, the streaming, time-dependent, and external source
components are formed as

.. math::

   \begin{aligned}
   \sig{*}\psi_{m}^{n+1} &= \left(\frac{2}{\gamma \, v \Delta t} + \sig{t} \right) \psi_{m}^{n+1} \\
   q_{m}^{*} &= q_{m}^{n+\gamma} + \frac{2}{\gamma \, v \Delta t} \psi^{n} + \sig{s} \, \phi^{n} + \chi \nu \sig{f} \, \phi^{n} + q_{m}^{n} - \bsOmega \cdot \nabla \psi^{n} - \sig{t} \psi^{n}_{m}
   \end{aligned}

which can be inserted into Equation :eq:`nte-spatial`.

With the flux known previously at time step :math:`n` and calculated at time step
:math:`n + \gamma`, the BDF2 step can take place to solve for the angular flux at
time step :math:`n + 1`. This is accomplished by taking the Taylor expansion of
Equation :eq:`nte-bdf2` and simplifying to get

.. math::

   \frac{1}{v} \frac{\partial \psi}{\partial t} = \frac{1}{v \Delta t} \left(\frac{2 - \gamma}{1 - \gamma} \psi^{n+1} - \frac{1}{\gamma (1 - \gamma)} \psi^{n+\gamma} + \frac{1 - \gamma}{\gamma} \psi^{n} \right) = \bA(\psi^{n+1}).

When converting it into a form ready for Equation :eq:`nte-spatial`, the terms are
combined as

.. math::

   \begin{aligned}
   \sig{*} \psi_{m}^{n+1} &= \left( \frac{2 - \gamma}{(1 - \gamma) \, v \Delta t} + \sig{t} \right) \psi_{m}^{n+1} \\
   q_{m}^{*} &= q_{m}^{n+1} + \frac{1}{\gamma \, (1 - \gamma) \, v \Delta t} \psi_{m}^{n+\gamma} - \frac{1 -\gamma}{\gamma \, v \Delta t} \psi_{m}^{n}
   \end{aligned}

to complete the BDF2 step for the TR-BDF2 temporal discretization. While it is
commonplace to use :math:`\gamma = 2 - \sqrt{2}` for the half step
value :cite:`dharmaraja2007`, in this analysis :math:`\gamma = 1/2` is used
according to :cite:`edwards2011`.
