.. _sec-nte-spatial:

One- and Two-Dimensional Spatial Discretization
===============================================

A standard sweeping method :cite:`bell1970nuclear` forms the core of the inner
iteration discussed in :ref:`the multigroup discrete ordinates method
<sec-nte-discrete>`. For this process, the angular flux is calculated at the cell
boundary and used to average the flux at the cell center. While there are a
number of different methods to discretize the spatial dimension for both one- and
two-dimensional geometries, this work focuses on the second order diamond
difference method :cite:`lewis1993`. The first step in this process is to break
down the streaming operator (:math:`\bsOmega \cdot \nabla \psi_{m, g}`) for
rectangular and spherical coordinates. From there, the diamond difference will be
used to solve for the flux at specific cell edges and average the neutron flux at
the cell center through

.. math::
   :label: nte-dd-1d

   \psi_{i} = \frac{1}{2} \left(\psi_{i+1/2} + \psi_{i-1/2} \right)

.. math::
   :label: nte-dd-2d-x

   \psi_{i,j} = \frac{1}{2} \left(\psi_{i+1/2,j} + \psi_{i-1/2,j} \right)

.. math::
   :label: nte-dd-2d-y

   \psi_{i,j} = \frac{1}{2} \left(\psi_{i,j+1/2} + \psi_{i,j-1/2} \right)

where :math:`i` and :math:`j` represent the :math:`x` and :math:`y` spatial
cells, respectively with :math:`i \in [1, \ldots, I]` and
:math:`j \in [1, \ldots, J]` and the :math:`\psi_{i \pm 1/2}` flux values
representing the spatial cell edges. If the incoming flux is known at
:math:`i - 1/2`, the average flux for each spatial cell can be solved through the
spatial sweep. There is a different spatial discretization method that will be
briefly used in a later verification chapter, which is a first order piece-wise
constant upwind scheme known as the step method. Unless noted otherwise, the
diamond difference is the spatial discretization scheme employed in this
analysis.

It should be noted that for clarity, fixed source problems will be discussed in
this section and both the time-dependent term and the :math:`g` subscripts will be
dropped from Equation :eq:`nte-discrete`. This alters the neutron transport
equation for a steady-state, mono-energetic problem as

.. math::
   :label: nte-spatial

   \bsOmega \cdot \nabla \psi_{m} + \sig{t} \psi_{m} = \sig{s} \phi + \chi \nu \sig{f} \phi + q_{m},

which will be the basis for the one- and two-dimensional spatial discretizations
of the streaming operator.


One-Dimensional Slab
--------------------

The simplest neutron transport is a one-dimensional slab, in which the streaming
operator is represented as

.. math::

   \bsOmega \cdot \nabla \psi_{m} = \mu_{m} \frac{\partial \psi_{m}}{\partial x}

where :math:`x` is the spatial dimension and :math:`\mu` is the angular
dimension. The partial derivative with respect to :math:`x` can be approximated
with the fundamental theorem of calculus by taking

.. math::
   :label: nte-calculus-x

   \frac{\partial \psi_{m}}{\partial x} = \frac{\psi_{i+1/2} - \psi_{i-1/2}}{x_{i+1/2} - x_{i-1/2}}

where the denominator can be simplified as
:math:`\Delta x_{i} = x_{i+1/2} - x_{i-1/2}`. Substituting this back into the
neutron transport equation, the equation becomes

.. math::

   \frac{\mu_{m}}{\Delta x_{i}} \left(\psi_{i+1/2} - \psi_{i-1/2}\right) + \sig{t} \psi_{i,m} = \sig{s} \phi_{i} + \chi \nu \sig{f} \phi_{i} + q_{i,m},

when solving for a specific spatial cell :math:`i` and angle :math:`\mu_{m}`.

The diamond difference method, defined in Equation :eq:`nte-dd-1d` can replace the
angular flux in the removal term in which the equation becomes

.. math::

   \frac{\mu_{m}}{\Delta x_{i}} \left(\psi_{i+1/2,m} - \psi_{i-1/2,m}\right) + \frac{\sig{t}}{2} (\psi_{i+1/2,m} + \psi_{i-1/2,m}) = \sig{s} \phi_{i} + \chi \nu \sig{f} \phi_{i} + q_{i,m}.

The sign of the angular dimension :math:`\mu` determines the angular flux to
solve for, in which the incoming boundary flux is known at :math:`\psi_{1/2,m}`
when :math:`\mu_{m} > 0` and likewise at :math:`\psi_{I+1/2,m}` when
:math:`\mu_{m} < 0`. In the instance where the angle :math:`\mu_{m} > 0`, the
angular flux at :math:`\psi_{i-1/2,m}` is known and the angular flux at
:math:`\psi_{i+1/2,m}` is calculated as

.. math::

   \psi_{i+1/2,m}^{\ell+1} \left( \frac{\mu_{m}}{\Delta x_{i}} + \frac{\sig{t}}{2} \right) = \sig{s} \phi_{i}^{\ell} + \chi \nu \sig{f} \phi_{i}^{\ell} + q_{i,m} + \psi_{i-1/2,m}^{\ell+1} \left( \frac{\mu_{m}}{\Delta x_{i}} - \frac{\sig{t}}{2} \right),

where the :math:`\ell` and :math:`\ell+1` are added to show an iterative approach
to solving this equation with the scalar flux being lagged by an iteration. The
scalar flux is updated according to Equation :eq:`nte-scalar`.


One-Dimensional Sphere
----------------------

The spatial discretization of the one-dimensional spherical geometry is more
involved, as the streaming operator becomes

.. math::
   :label: nte-sphere-initial

   \bsOmega \cdot \nabla \psi_{m} = \frac{\mu_{m}}{x^2} \frac{\partial}{\partial x}\left(x^2 \psi_{m} \right) + \frac{1}{x} \frac{\partial}{\partial \mu} \left(\left[1 - \mu^2\right] \psi_{m} \right),

where :math:`x` is the spatial dimension and :math:`\mu` is the angular
dimension. Following :cite:`lewis1993`, angular differencing coefficients
:math:`\alpha` are added to the transport equation to become

.. math::
   :label: nte-sphere-angle-1

   \begin{aligned}
   \frac{\mu_{m}}{x^2} \frac{\partial}{\partial x} x^2 \psi_{m}(x) &+ \frac{1}{x \, w_{m}} \left[ \alpha_{m+1/2} \psi_{m+1/2} (x) - \alpha_{m-1/2} \psi_{m-1/2} (x) \right] \\
   &\hspace{50pt} + \sig{t} \psi_{m} = \sig{s} \phi + \chi \nu \sig{f} \phi + q_{m},
   \end{aligned}

where :math:`w_{m}` is the discrete ordinates weight for angle :math:`m`. This
introduces the half angle :math:`m \pm 1/2` term where

.. math::
   :label: nte-sphere-m-diamond-1

   \psi_{m}(x) = \frac{1}{2} \left( \psi_{m-1/2}(x) + \psi_{m+1/2}(x) \right)

is the diamond difference method in the angular dimension. The half angle is
calculated as

.. math::
   :label: nte-sphere-m-update

   \mu_{m+1/2} = \mu_{m-1/2} + 2 w_{m}

and the starting and ending directions are :math:`\mu_{1/2} = -1` and
:math:`\mu_{M+1/2} = 1`. For the streaming term to vanish in a uniform flux
problem, the angular differencing coefficient must have the relationship

.. math::
   :label: nte-sphere-alpha-update

   \alpha_{m+1/2} = \alpha_{m-1/2} - \mu_{m} w_{m}

and the property :math:`\alpha_{1/2} = \alpha_{M+1/2} = 0` to satisfy the neutron
balance condition in spherical geometry. Substituting Equation
:eq:`nte-sphere-m-diamond-1` into Equation :eq:`nte-sphere-angle-1` for the
:math:`\psi_{m+1/2}(x)` term, the transport equation becomes

.. math::
   :label: nte-sphere-angle-2

   \begin{aligned}
   \frac{\mu_{m}}{x^2} \frac{\partial}{\partial x} x^2 \psi_{m}(x) &+ \frac{1}{x \, w_{m}} \left[ 2 \alpha_{m+1/2} \psi_{m} (x) - (\alpha_{m-1/2} + \alpha_{m+1/2}) \psi_{m-1/2} (x) \right] \\
   &\hspace{50pt} + \sig{t} \psi_{m} = \sig{s} \phi + \chi \nu \sig{f} \phi + q_{m},
   \end{aligned}

where :math:`\psi_{m}(x)` can be solved when :math:`\psi_{m-1/2}(x)` is known.

Before proceeding with the spatial sweep, the partial derivative with respect to
:math:`x` in Equation :eq:`nte-sphere-angle-2` must be handled appropriately in
curvilinear coordinates. For this, the volume :math:`V_{i}` of one spherical
shell is

.. math::
   :label: nte-sphere-volume

   V_{i} = \frac{4 \pi}{3} \left( x^3_{i+1/2} - x^3_{i-1/2} \right),

where :math:`i \pm 1/2` are the cell edges and the surface area at the cell
boundary is :math:`A_{i+1/2} = 4 \, \pi \, x^2_{i+1/2}`. Integrating Equation
:eq:`nte-sphere-angle-2` by Equation :eq:`nte-sphere-volume` and dividing by the
volume, the spherical neutron transport equation becomes

.. math::
   :label: nte-sphere-angle-3

   \begin{aligned}
   &\frac{\mu_{m}}{V_{i}} \left[ A_{i+1/2} \psi_{m,i+1/2} - A_{i-1/2} \psi_{m,i-1/2} \right] \\
   &\hspace{30pt} + \frac{\left( A_{i+1/2} - A_{i-1/2} \right)}{V_{i} \, w_{m}} \left[ 2 \alpha_{m+1/2} \psi_{m,i} - \left( \alpha_{m+1/2} + \alpha_{m-1/2} \right) \psi_{m-1/2,i} \right] \\
   &\hspace{60pt} + \sig{t} \psi_{m,i} = \sig{s} \phi_{i} + \chi \nu \sig{f} \phi_{i} + q_{m,i},
   \end{aligned}

for a given angle :math:`m` and spatial cell :math:`i`.

With Equation :eq:`nte-sphere-angle-3`, the half angular flux is required to
calculate the flux at a given angle :math:`m` and cell :math:`i`. To do this, the
angular derivative in Equation :eq:`nte-sphere-initial` can be used with the
starting direction :math:`\mu_{1/2} = -1` as the inward most direction to cancel
out the angular derivative. The neutron transport equation for spherical geometry
would simplify to

.. math::

   \mu_{1/2} \, \frac{\partial}{\partial x} \psi_{1/2}(x) + \sig{t} \psi_{1/2} = \sig{s} \phi + \chi \nu \sig{f} \phi + q,

which can be rearranged to solve for the starting angular flux
:math:`\psi_{1/2,i}` as

.. math::
   :label: nte-sphere-half-angle

   \psi_{1/2,i} = \frac{2 \psi_{1/2,i+1/2} + \Delta x_{i} \left(\sig{s} \phi_{i} + \chi \nu \sig{f} \phi_{i} + q_{m,i} \right)}{2 + \Delta x_{i} \, \sig{t}}

with :math:`\Delta x_{i} = x_{i+1/2} - x_{i-1/2}`.

After the flux for the half angle :math:`m = 1/2` is calculated, Equation
:eq:`nte-sphere-angle-3` can be rearranged to solve for the cell flux center as

.. math::
   :label: nte-sphere-sweep

   \psi_{m,i} = \frac{a_{m,i} + b_{m,i} + c_{m,i}}{d_{m,i}}

for :math:`\mu < 0` with

.. math::
   :label: nte-sphere-sweep-neg

   \begin{aligned}
   a_{m,i} &= |\mu_{m}| (A_{i+1/2} + A_{i-1/2}) \, \psi_{m,i+1/2} \\
   b_{m,i} &= \frac{1}{w_{m}} (A_{i+1/2} - A_{i-1/2}) (\alpha_{m+1/2} + \alpha_{m-1/2}) \,\psi_{m-1/2,i} \\
   c_{m,i} &= V_{i} \left(\sig{s} \phi_{i} + \chi \nu \sig{f} \phi_{i} + q_{m,i} \right) \\
   d_{m,i} &= 2 |\mu_{m}| A_{i-1/2} + \frac{2}{w_{m}}(A_{i+1/2} - A_{i-1/2}) \, \alpha_{m+1/2} + V_{i} \, \sig{t}
   \end{aligned}

where flux at :math:`\psi_{m,i+1/2}` is known. The :math:`\psi_{m,i-1/2}` is
updated using the diamond difference method as

.. math::
   :label: nte-sphere-x-diamond

   \psi_{m,i-1/2} = 2 \, \psi_{m,i} - \psi_{m,i+1/2}

and the angular flux half angle :math:`\psi_{m+1/2,i}` can be updated using
Equation :eq:`nte-sphere-m-diamond-1`. If sweeping from the center of the sphere
to the edge, :math:`\mu > 0`, the :math:`a_{m,i}` and :math:`d_{m,i}` equations in
Equation :eq:`nte-sphere-sweep-neg` are updated as

.. math::
   :label: nte-sphere-sweep-pos

   \begin{aligned}
   a_{m,i} &= |\mu_{m}| (A_{i+1/2} + A_{i-1/2}) \, \psi_{m,i-1/2} \\
   d_{m,i} &= 2 |\mu_{m}| A_{i+1/2} + \frac{2}{w_{m}}(A_{i+1/2} - A_{i-1/2}) \, \alpha_{m+1/2} + V_{i} \, \sig{t},
   \end{aligned}

where the angular flux :math:`\psi_{m, i+1/2}` is updated using the diamond
difference method.

The center of the sphere can approximated as

.. math::

   \psi_{M+1-m, 1/2} = \psi_{m,1/2}

and treated as a reflective slab boundary condition. This however introduces a
nonphysical dip in the spatial flux as pointed out in :cite:`lewis1993` and it
would be more appropriate to approximate the flux as

.. math::
   :label: nte-sphere-center

   \psi_{m-1/2, 1/2} = \psi_{m+1/2,1/2}

where the starting flux is calculated using Equation :eq:`nte-sphere-half-angle`
at the origin for all angles. This can be combined with a weighted diamond
difference scheme as presented in :cite:`morel1984` to update the half angle as

.. math::
   :label: nte-sphere-m-diamond-2

   \psi_{m,i} = (1 - \tau) \, \psi_{m-1/2,i} + \tau \, \psi_{m+1/2,i}

instead of using Equation :eq:`nte-sphere-m-diamond-1`. The weight :math:`\tau` is

.. math::

   \tau = \frac{\mu_{m} - \mu_{m-1/2}}{\mu_{m+1/2} - \mu_{m-1/2}}

where the :math:`\mu_{m \pm 1/2}` are updated according to Equation
:eq:`nte-sphere-m-update`.

All the equations are presented to solve for the one-dimensional sphere problem
using discrete ordinates and the diamond difference spatial discretization.
Assuming the angular dimension is traversed from :math:`\mu = -1` to
:math:`\mu = 1`, the initial conditions are :math:`\mu_{1/2} = -1` and
:math:`\alpha_{1/2} = 0`, which can solve for the half angle
:math:`\psi_{1/2,i}` using Equation :eq:`nte-sphere-half-angle`. The angular flux
at the spatial cell centers using Equation :eq:`nte-sphere-sweep-neg`, where the
spatial cell edge is updated using Equation :eq:`nte-sphere-x-diamond` and the
angle edge is updated using Equation :eq:`nte-sphere-m-diamond-2`. The angular
differencing coefficient and half angle term are updated using Equations
:eq:`nte-sphere-alpha-update` and :eq:`nte-sphere-m-update`. When the spatial
sweep is from the spherical center to the edge :math:`\mu > 0`, the half angle
Equation :eq:`nte-sphere-center` is used for the initialization and the
:math:`a_{m,i}` and :math:`d_{m,i}` terms in Equation :eq:`nte-sphere-sweep` are
updated with Equation :eq:`nte-sphere-sweep-pos`.


Two-Dimensional Slab
--------------------

The streaming term in two dimensions would result in

.. math::

   \bsOmega \cdot \nabla \psi_{m} = \mu_{m} \frac{\partial \psi_{m}}{\partial x} + \eta_{m} \frac{\partial \psi_{m}}{\partial y}

where :math:`\mu` and :math:`\eta` are the angular dimensions and :math:`x` and
:math:`y` are the spatial dimensions. As with the one-dimensional slab geometry,
the fundamental theorem of calculus can be used to discretize the partial
derivative with respect to :math:`y` as

.. math::

   \frac{\partial \psi_{m}}{\partial y} = \frac{\psi_{j+1/2} - \psi_{j-1/2}}{y_{j+1/2} - y_{j-1/2}}

with the derivative for :math:`x` shown in Equation :eq:`nte-calculus-x` and
:math:`\Delta y_{j} = y_{j+1/2} - y_{j-1/2}` equaling the cell width in the
:math:`y` direction. Inserting these terms back into the neutron transport
equation, the resulting equation is

.. math::

   \begin{aligned}
   \frac{\mu_{m}}{\Delta x_{i}} \left(\psi_{i+1/2,j,m} - \psi_{i-1/2,j,m} \right) &+  \frac{\eta_{m}}{\Delta y_{j}} \left(\psi_{i,j+1/2,m} - \psi_{i,j-1/2,m} \right) + \sig{t} \psi_{i,j,m} \\
   &= \sig{s} \phi_{i,j} + \chi \nu \sig{f} \phi_{i,j} + q_{i,j,m}
   \end{aligned}

when solving for a specific spatial cell :math:`(i, j)` and angle
:math:`\mu_{m}, \eta_{m}`.

For the specific discrete ordinate, the same principle used for the known flux in
the one-dimensional slab problem can be expanded to the two-dimensional slab.
Assuming that the discrete ordinate results in :math:`\mu_{m} > 0` and
:math:`\eta_{m} > 0`, the fluxes at the boundaries :math:`\psi_{i-1/2,j,m}` and
:math:`\psi_{i,j-1/2,m}` will be known. The diamond difference method in
Equations :eq:`nte-dd-2d-x` and :eq:`nte-dd-2d-y` can be rearranged to solve for
the opposite sides, namely :math:`\psi_{i+1/2,j,m}` and :math:`\psi_{i,j+1/2,m}`.
These terms are reinserted into the transport equation in which only the angular
flux at the cell center :math:`\psi_{i,j,m}` is unknown. Combining like terms, the
two-dimensional transport sweep equation is

.. math::

   \begin{aligned}
   &\left(\sig{t} + \frac{2 \mu_{m}}{\Delta x_{i}} + \frac{2 \eta_{m}}{\Delta y_{j}} \right) \psi_{i,j,m} \\
   &= \sig{s} \phi_{i,j} + \chi \nu \sig{f} \phi_{i,j} + q_{i,j,m} + \frac{2 \mu_{m}}{\Delta x_{i}} \psi_{i-1/2,j,m} + \frac{2 \eta_{m}}{\Delta y_{j}} \psi_{i,j-1/2,m},
   \end{aligned}

which can be solved in the same iterative scheme as the one-dimensional slab case.
The angular flux at the cell edges, :math:`\psi_{i+1/2,j,m}` and
:math:`\psi_{i,j+1/2,m}`, are updated using the diamond difference Equations
:eq:`nte-dd-2d-x` and :eq:`nte-dd-2d-y` before proceeding to the next spatial
cell.
