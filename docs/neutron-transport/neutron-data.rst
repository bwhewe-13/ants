Multigroup Neutron Data
===============================================

This discusses the formation of the neutron data used in ANTS, as well as
different coarsening processes for the total, scattering, and fission data.

Neutron Velocity
----------------

The neutron velocity :math:`v` was calculated from the energy grid at the group centers :math:`E` using the relativistic speed. The constants (neutron rest mass 
:math:`m` and speed of light :math:`c`) are found in the 
``ants.constants.py`` file.

The classical energy to velocity equation is :math:`E = 0.5 m v^2`. However, 
since there are energies close to the speed of light, the relativistic 
equation  and derivation is accomplished below.
The original equation, from :cite:`bertozzi1964speed`, is

.. math::
   :label: velocity-01

   \begin{equation} 
      E = m c^2 \left( \frac{1}{\sqrt{1 - \frac{v^2}{c^2}}} - 1 \right)
   \end{equation}

which can be reformed as 

.. math::
   :label: velocity-02

   \begin{equation} 
      \frac{E}{m c^2} + 1 = \frac{1}{\sqrt{1 - \frac{v^2}{c^2}}}.
   \end{equation}

Setting the left hand side to :math:`\gamma`, we can solve for :math:`v`
as

.. math::
   :label: velocity-03

   \begin{equation} \begin{split}
      \frac{1}{\sqrt{1 - \frac{v^2}{c^2}}} &= \gamma \\
      1 - \frac{v^2}{c^2} &= \frac{1}{\gamma^2} \\
      v^2 &= -\frac{c^2}{\gamma^2} + c^2 \\
      v &= \sqrt{c^2 -\frac{c^2}{\gamma^2}}
   \end{split} \end{equation}

This is simplified so

.. math::
   :label: velocity-04

   \begin{equation}
      v = \frac{c}{\gamma} \sqrt{c^2 - 1} \qquad \text{where} \qquad 
      \gamma = \frac{E}{m c^2} + 1
   \end{equation}


.. bibliography::
