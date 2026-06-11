.. _sec-mnp-residual:

Curve Fit Residual Formulation
==============================

The analytical curve fit solutions are created for each material, angle
:math:`m`, and energy group :math:`g` to approximate the numerical flux
:math:`\psi_{m,g}`. The residual between the curve fit solutions and the
numerical flux is calculated analytically as well. To calculate the continuous
residual :math:`R_{m,g}(\bx)` of the curve fit flux, the calculated spline
:math:`S_{m,g}(\bx)` and :math:`\overline{S}_{g}(\bx)` replace the angular
:math:`\psi_{m,g}(\bx)` and scalar flux :math:`\phi_{g}(\bx)` in
:eq:`mnp-discrete`, respectively. The discrete residual :math:`r_{i,m,g}` for
one-dimensional problems is calculated for each cell by integrating the splines
over the cell edges in the transport equation. Since the spline knots are located
at the centers of the spatial cells, in order to integrate the residual for a
specific cell, the integration of two splines is required. It should be noted that
the spatial cell edges are represented with :math:`i \pm 1/2` and
:math:`j \pm 1/2` subscripts, while the spline :math:`S_{i}(\bx)` uses
:math:`[i, i+1]` for its interval.


One-Dimensional Spline Residual Integration
-------------------------------------------

The discrete ordinates multigroup neutron transport equation in :eq:`nte-discrete`
can be rearranged for a steady-state one-dimensional slab as

.. math::

   \mu_{m} \frac{d}{d x} \psi_{m,g}(x) + \sig[g]{t} \psi_{m,g}(x)
   = \sum_{g'=1}^{G} \sig[g' \rightarrow g]{s} \phi_{m,g}(x) + q_{m,g}.

This is further rearranged into a residual problem, where the continuous residual
:math:`R_{m,g}(x)` is solved for as

.. math::

   R_{m,g}(x) = \mu_{m} \frac{d}{d x} S_{m,g}(x) + \sig[g]{t} S_{m,g}(x)
   - \sum_{g'=1}^{G} \sig[g' \rightarrow g]{s} \overline{S}_{m,g}(x) - q_{m,g},

where the curve fit flux is substituted for the angular flux. The discrete
residual for spatial cell :math:`i` is taken by integrating over the two halves
of the spatial cell. Spline :math:`S_{i-1}(x)` is integrated from
:math:`x_{i - 1/2}` to :math:`x_{i}` and spline :math:`S_{i}(x)` is integrated
from :math:`x_{i}` to :math:`x_{i+1/2}`. The resulting transport equation is

.. math::

   \begin{aligned}
   r_{i,m,g} &= \mu_{m} \int_{x_{i-1/2}}^{x_{i}} dx \, \left( \frac{d}{d x} S_{i-1,m,g}(x) \right) + \mu_{m} \int_{x_{i}}^{x_{i+1/2}} dx \, \left( \frac{d}{d x} S_{i, m,g}(x) \right) \\
   &\quad + \sig[g]{t} \int_{x_{i-1/2}}^{x_{i}} dx \, S_{i-1, m,g}(x) + \sig[g]{t} \int_{x_{i}}^{x_{i+1/2}} dx \, S_{i, m,g}(x) \\
   &\quad - \sum_{g'=1}^{G} \sig[g' \rightarrow g]{s} \int_{x_{i-1/2}}^{x_{i}} dx \, \overline{S}_{i-1,m,g}(x) - \sum_{g'=1}^{G} \sig[g' \rightarrow g]{s} \int_{x_{i}}^{x_{i+1/2}} dx \, \overline{S}_{i,m,g}(x) \\
   &\quad - \int_{x_{i-1/2}}^{x_{i+1/2}} dx \, q_{i,m,g}
   \end{aligned}

for the discrete residual at spatial cell :math:`i \in [1, I]`. The integrals can
be calculated using either :eq:`mnp-cubic-int-02` and :eq:`mnp-cubic-dx-int-02`
for the cubic Hermite splines or :eq:`mnp-quintic-int-02` and
:eq:`mnp-quintic-dx-int-02` for the quintic Hermite splines. At the endpoints
where :math:`i = 1, I` and at material interfaces, a single spline is used for the
integration of the spatial cell.

The current process for discretizing the residual requires the use of integrating
two splines for each half cell, which requires more computational work. It would
be simpler for the residual calculation if the curve fit splines used the flux at
the cell edges :math:`x_{i \pm 1/2}` for the knot locations instead of the cell
centers. This has been investigated for one-dimensional problems but does not show
a significant improvement in the spatial discretization error estimation. The
utilization of cell edges as spline knots was not used in the final results.


Two-Dimensional Spline Residual Integration
-------------------------------------------

The continuous residual :math:`R_{m,g}(x, y)` can also be calculated using the
two-dimensional neutron transport equation in :eq:`nte-discrete`. Rearranging the
transport equation for a steady-state two-dimensional slab, the neutron equation
becomes

.. math::

   \begin{aligned}
   \mu_{m} \frac{d}{d x} \psi_{m,g}(x, y) + \eta_{m} \frac{d}{d y} \psi_{m,g}(x, y)
   &+ \sig[g]{t} \psi_{m,g}(x, y) \\
   &= \sum_{g'=1}^{G} \sig[g' \rightarrow g]{s} \phi_{m,g}(x, y) + q_{m,g}.
   \end{aligned}

Adjusting this equation for the continuous residual problem, the transport
equation is

.. math::

   \begin{aligned}
   R_{m,g}(x, y) &= \mu_{m} \frac{d}{d x} S_{m,g}(x, y) + \eta_{m} \frac{d}{d y} \psi_{m,g}(x, y) + \sig[g]{t} \psi_{m,g}(x, y) \\
   &\quad - \sum_{g'=1}^{G} \sig[g' \rightarrow g]{s} \overline{S}_{m,g}(x, y) - q_{m,g}
   \end{aligned}

where the curve fit flux :math:`S_{m,g}(x, y)` replaces the angular flux
:math:`\psi_{m,g}(x,y)`. The discrete residual for spatial cell
:math:`i \in [1, I]`, :math:`j \in [1, J]`, is determined by taking a double
integral across the cell edges :math:`[x_{i-1/2}, x_{i+1/2}]` and
:math:`[y_{j-1/2}, y_{j+1/2}]`. Taking the integrals results in the neutron
transport equation becoming

.. math::
   :label: mnp-2d-residual

   \begin{aligned}
   r_{i,j,m,g} &= \mu_{m} \int dy \int dx \left(\frac{d}{d x} \psi_{m,g}(x, y) \right) + \eta_{m} \int dy \int dx \left(\frac{d}{d y} \psi_{m,g}(x, y) \right) \\
   &\quad + \sig[g]{t} \int dy \int dx \, S_{m,g}(x, y) \\
   &\quad - \sum_{g'=1}^{G} \sig[g' \rightarrow g]{s} \int dy \int dx \, \overline{S}_{m,g}(x, y) - \int dy \int dx \, q_{m,g},
   \end{aligned}

which must take the integrals of four splines. The limits of the integrals are
shown as

.. math::

   \begin{aligned}
   \int dy \int dx \, S(x, y) &:= \int_{y_{j-1/2}}^{y_{j}} dy \int_{x_{i-1/2}}^{x_{i}} dx \, S_{i-1,j-1}(x, y) \\
   &\quad + \int_{y_{j-1/2}}^{y_{j}} dy \int_{x_{i}}^{x_{i+1/2}} dx \, S_{i,j-1}(x, y) \\
   &\quad + \int_{y_{j}}^{y_{j+1/2}} dy \int_{x_{i-1/2}}^{x_{i}} dx \, S_{i-1,j}(x, y) \\
   &\quad + \int_{y_{j}}^{y_{j+1/2}} dy \int_{x_{i}}^{x_{i+1/2}} dx \, S_{i,j}(x, y)
   \end{aligned}

for the spatial cell with the bounds :math:`i \pm 1/2` and :math:`j \pm 1/2`.
These limits can be used for all the terms in :eq:`mnp-2d-residual` and are
visually shown for the :ref:`center four-spline case <fig-mnp-2d-integral-04>`.
The integrals can be calculated using either :eq:`mnp-cubic-int-01` and
:eq:`mnp-cubic-dx-int-01` for the cubic Hermite splines or
:eq:`mnp-quintic-int-01` and :eq:`mnp-quintic-dx-int-01` for the quintic Hermite
splines.

Expanding into higher spatial dimensions results in additional cases when dealing
with material boundaries. While the interior points integrate over four splines,
there are instances where there is only one or two splines interpolating on a
spatial cell. The first case is the material corner case, which uses one spline,
while the material edge case uses two splines. The integration limits for the
spatial cell in :math:`[i-1/2, i+1/2]` and :math:`[j-1/2, j+1/2]` with a material
corner is

.. math::

   \int dy \int dx \, S(x, y) := \int_{y_{j-1/2}}^{y_{j+1/2}} dy \int_{x_{i-1/2}}^{x_{i+1/2}} dx \, S_{i,j}(x, y),

and is shown for the :ref:`corner single-spline case <fig-mnp-2d-integral-01>`. If
the same spatial cell (:math:`[i-1/2, i+1/2]` and :math:`[j-1/2, j+1/2]`) had a
material edge at :math:`x_{i-1/2}`, the integration limits would be

.. math::

   \begin{aligned}
   \int dy \int dx \, S(x, y) &:= \int_{y_{j-1/2}}^{y_{j+1/2}} dy \int_{x_{i-1/2}}^{x_{i}} dx \, S_{i-1,j}(x, y) \\
   &\quad + \int_{y_{j-1/2}}^{y_{j+1/2}} dy \int_{x_{i}}^{x_{i+1/2}} dx \, S_{i,j}(x, y),
   \end{aligned}

which divides the spatial cell in half vertically and integrates each side with
its appropriate spline. This is demonstrated for the :ref:`edge two-spline case
<fig-mnp-2d-integral-02>`. To limit the residual to these three cases, a region
splitting method is used where rectangular regions are created for the different
materials. There are instances where regions might have the same material as an
adjacent region but are separated to maintain their rectangular shape. This is
demonstrated in the two-dimensional results section of this chapter.

The three spline integration scenarios are illustrated below. Each compares a
different case that can occur: the corner single-spline integral, the edge
two-spline integral, and the center four-spline integral. The spline knots are the
known fluxes while the cell edges are the corners of each cell.

.. _fig-mnp-2d-integral-01:

.. tikz::
   :libs: decorations.pathreplacing
   :include: tikz/integral-01-splines.tikz

\(a) The corner single-spline integral.

.. _fig-mnp-2d-integral-02:

.. tikz::
   :libs: decorations.pathreplacing
   :include: tikz/integral-02-splines.tikz

\(b) The edge two-spline integral.

.. _fig-mnp-2d-integral-04:

.. tikz::
   :libs: decorations.pathreplacing
   :include: tikz/integral-04-splines.tikz

\(c) The center four-spline integral.

While it would have been possible to set the spatial cell corners as the spline
knots and integrate over one spline for each cell, additional approximations are
needed. In two-dimensional space, the incoming flux is known at the interface and
approximated at the midpoint of the edge, i.e. :math:`\psi_{i,j-1/2}` for a source
entering from the left edge. The four interfaces are needed to approximate the
edge values, :math:`\psi_{i \pm 1/2, j \pm 1/2}`, which can then be set as the
spline knots. This has been explored but it has not yielded beneficial results for
identifying the spatial discretization error.
