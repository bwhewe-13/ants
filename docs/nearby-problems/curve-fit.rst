.. _sec-mnp-curve-fit:

Hermite Spline Curve Fit Interpolation
======================================

The curve fit solution creates an analytical solution of the angular flux for
each energy group :math:`g` and angle :math:`m`. The analytical curve fit
solutions are represented as :math:`S_{m,g}(\bx)` where the subscripts will be
dropped for simplicity and the individual energy group and angular direction
will be implied. It should be noted that while there has been research into the
optimal placement of the spline knots :cite:`idais2019`, for this analysis they
are placed at the cell centers, where the angular flux is approximated as seen in
the :ref:`spatial discretization section <sec-nte-spatial>`. This section
demonstrates the formulation of the cubic and quintic Hermite splines as shown
for one-dimensional problems. These splines are expanded to two dimensions using
the bicubic and biquintic interpolations, respectively.


.. _sec-mnp-derivative:

Finite Difference for Numerical Differentiation
-----------------------------------------------

The cubic and quintic Hermite splines require the use of the first and second
derivative of the angular flux at the knot points. As the numerical angular flux
is approximated using source iteration, these derivatives must be approximated.
For this analysis, the Taylor expansion is used to estimate the second order
derivatives for non-uniform spatial grids. Second order
:math:`\mathcal{O}(\Delta x^2)` derivatives were chosen for the numerical
differentiation to allow for better accuracy when applying it to the spline
interpolation. The Taylor series is represented as

.. math::

   \psi(x_{i}) = \sum_{n = 0}^{\infty} \frac{\psi^{(n)}(x)}{n!} (x_{i} - x)^{n}

for point :math:`x_{i}` for a specific energy group and angle. This is expanded
for the grid points around :math:`x_{i}`, resulting in

.. math::
   :label: mnp-taylor-01

   \psi(x_{i-2}) = \psi(x_{i}) - (x_{i} - x_{i-2})\; \psi'(x_{i})
   + \frac{1}{2} (x_{i} - x_{i-2})^2 \; \psi''(x_{i})

.. math::
   :label: mnp-taylor-02

   \psi(x_{i-1}) = \psi(x_{i}) - (x_{i} - x_{i-1})\; \psi'(x_{i})
   + \frac{1}{2} (x_{i} - x_{i-1})^2 \; \psi''(x_{i})

.. math::
   :label: mnp-taylor-03

   \psi(x_{i+1}) = \psi(x_{i}) + (x_{i+1} - x_{i})\; \psi'(x_{i})
   + \frac{1}{2} (x_{i+1} - x_{i})^2 \; \psi''(x_{i})

.. math::
   :label: mnp-taylor-04

   \psi(x_{i+2}) = \psi(x_{i}) + (x_{i+2} - x_{i})\; \psi'(x_{i})
   + \frac{1}{2} (x_{i+2} - x_{i})^2 \; \psi''(x_{i})

where :math:`\psi'` and :math:`\psi''` are the first and second derivatives with
respect to :math:`x`, respectively. The forward and backward differences are used
for the endpoints and the central difference is used for the midpoints. It is
important that there are at least three points
:math:`(x_{i-1}, x_{i}, x_{i+1})` for these calculations, which is required for
each material zone when calculating the curve fit flux. The formulas for
calculating the first and second derivatives are presented below.


Forward Difference Formula
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the endpoint at :math:`i = 1`, the forward difference formula is used with
the :math:`\psi(x_{i+1})` and :math:`\psi(x_{i+2})` Taylor expansions in
:eq:`mnp-taylor-03` and :eq:`mnp-taylor-04`. Combining these equations and
eliminating the :math:`\psi''(x_{i})` term, the first derivative becomes

.. math::

   \psi'(x_{i}) = \frac{\psi(x_{i}) - \psi(x_{i+1})}{x_{i} - x_{i+1}}
   + \frac{\psi(x_{i}) - \psi(x_{i+2})}{x_{i} - x_{i+2}}
   + \frac{-\psi(x_{i+1}) + \psi(x_{i+2})}{x_{i+1} - x_{i+2}}

for a non-uniform spatial grid. For the second derivative, the same equations are
used to eliminate the :math:`\psi'(x_{i})` term resulting in the solution

.. math::

   \psi''(x_{i}) = \frac{2 \, \psi(x_{i})}{(x_{i+1} - x_{i}) (x_{i+2} - x_{i})}
   + \frac{2 \, \psi(x_{i+1})}{(x_{i+1} - x_{i}) (x_{i+1} - x_{i+2})}
   + \frac{2 \, \psi(x_{i+2})}{(x_{i+2} - x_{i})(x_{i+2} - x_{i+1})},

which is second order accurate.


Backward Difference Formula
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the endpoint at :math:`i = I`, the second order backward difference formula
is used to calculate the first and second derivatives numerically. The
:math:`\psi(x_{i-2})` and :math:`\psi(x_{i-1})` Taylor expansions in
:eq:`mnp-taylor-01` and :eq:`mnp-taylor-02` are used to eliminate the
:math:`\psi''(x_{i})` term for the first derivative. This results in

.. math::

   \psi'(x_{i}) = \frac{\psi(x_{i}) - \psi(x_{i-1})}{x_{i} - x_{i-1}}
   + \frac{\psi(x_{i}) - \psi(x_{i-2})}{x_{i} - x_{i-2}}
   + \frac{-\psi(x_{i-1}) + \psi(x_{i-2})}{x_{i-1} - x_{i-2}}

for a non-uniform spatial grid. The second derivative with respect to :math:`x`
uses the same equations and eliminates the :math:`\psi'(x_{i})` term, resulting
in

.. math::

   \psi''(x_{i}) = \frac{2 \, \psi(x_{i})}{(x_{i} - x_{i-1}) (x_{i} - x_{i-2})}
   + \frac{2 \, \psi(x_{i-1})}{(x_{i-2} - x_{i-1}) (x_{i} - x_{i-1})}
   + \frac{2 \, \psi(x_{i-2})}{(x_{i} - x_{i-2})(x_{i-1} - x_{i-2})}

to maintain second order accuracy.


Central Difference Formula
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The central difference method is used for all the midpoints of the problem. The
first derivative uses the :math:`\psi(x_{i-1})` and :math:`\psi(x_{i+1})` Taylor
expansions in :eq:`mnp-taylor-02` and :eq:`mnp-taylor-03`. Eliminating the
:math:`\psi''(x_{i})` term results in

.. math::

   \psi'(x_{i}) = \frac{\psi(x_{i}) - \psi(x_{i-1})}{x_{i} - x_{i-1}}
   + \frac{\psi(x_{i}) - \psi(x_{i+1})}{x_{i} - x_{i+1}}
   + \frac{-\psi(x_{i-1}) + \psi(x_{i+1})}{x_{i-1} - x_{i+1}}

for a non-uniform spatial grid. Likewise, the second derivative uses the same
equations and eliminates the :math:`\psi'(x_{i})` term, which simplifies to

.. math::

   \psi''(x_{i}) = \frac{2 \, \psi(x_{i-1})}{(x_{i+1} - x_{i-1}) (x_{i} - x_{i-1})}
   + \frac{2 \, \psi(x_{i})}{(x_{i} - x_{i-1}) (x_{i} - x_{i+1})}
   + \frac{2 \, \psi(x_{i+1})}{(x_{i+1} - x_{i-1})(x_{i+1} - x_{i})}

for second order accuracy.


Cubic Hermite Splines
---------------------

A cubic Hermite spline :math:`S_{i}(x)` can be constructed from the general third
order polynomial

.. math::

   S_{i}(x) = a_{i} + b_{i} (x - x_{i}) + c_{i} (x - x_{i})^2 + d_{i} (x - x_{i})^3

where :math:`a_{i}`, :math:`b_{i}`, :math:`c_{i}`, and :math:`d_{i}` are the
coefficients to solve for and :math:`i \in \{1, I-1 \}` number of splines
:cite:`roy2007`. The interval :math:`[x_{i}, x_{i+1}]` is chosen for spline
:math:`S_{i}(x)` where :math:`x_{i}` and :math:`x_{i+1}` are the knot points.
Using the constraints

.. math::

   \begin{aligned}
   S_{i}(x_{i}) &= \psi_{i-1} \\
   S_{i}(x_{i+1}) &= \psi_{i+1} \\
   \partial_{x} S_{i}(x_{i}) &= \partial_{x} \psi_{i} \\
   \partial_{x} S_{i}(x_{i+1}) &= \partial_{x} \psi_{i+1},
   \end{aligned}

the spline can be converted into the form

.. math::

   S_i(x) = \psi_{i} \, h_{0,0}(x) + \psi_{i+1} \, h_{1,0}(x)
   + \partial_{x} (\psi_{i}) \, h_{0,1}(x) + \partial_{x} (\psi_{i+1}) \, h_{1,1}(x),

where :math:`h_{0,0}(x)`, :math:`h_{1,0}(x)`, :math:`h_{0,1}(x)`, and
:math:`h_{1,1}(x)` are the basis functions. For ease, :math:`\tx` is formulated to
normalize the point :math:`x` on the given interval :math:`[x_{i}, x_{i+1}]` as

.. math::
   :label: mnp-interp-tx

   \tx = \frac{x - x_{i}}{x_{i+1} - x_{i}},

which can be used to express the basis functions as

.. math::

   \begin{aligned}
   h_{0,0}(x) &= 2 \tx^3 - 3 \tx^2 + 1 \\
   h_{1,0}(x) &= -2 \tx^3 + 3 \tx^2 \\
   h_{0,1}(x) &= \Delta_{i} (\tx^3 - 2 \tx^2 + \tx) \\
   h_{1,1}(x) &= \Delta_{i} (\tx^3 - \tx^2).
   \end{aligned}

The width of the interval is represented as :math:`\Delta_{i} = x_{i+1} - x_{i}`.

The spline equation is converted to matrix form, which will be easier when moving
to bicubic interpolation in two dimensions. This results in
:math:`S_{i}(x) = \bX \, \bB \, \bu` where :math:`\bX` is the input matrix,
:math:`\bB` is the basis matrix, and :math:`\bu` is the control vector. The input
matrix is represented as :math:`[1, \; \tx, \; \tx^2, \; \tx^3]` for each row of
the input values, while the control vector is
:math:`[\psi_{i}, \; \psi_{i+1}, \; \Delta_{i} \, \partial_x \psi_{i}, \; \Delta_{i} \, \partial_x \psi_{i+1}]^T`.
Lastly, the basis matrix is

.. math::
   :label: mnp-cubic-basis

   \bB =
   \begin{bmatrix}
   1 & 0 & 0 & 0 \\
   0 & 0 & 1 & 0 \\
   -3 & 3 & -2 & -1 \\
   2 & -2 & 1 & 1
   \end{bmatrix}

which can be used to formulate the spline :math:`S_{i}(x)`.

When using the cubic Hermite spline in the neutron transport equation, the
derivatives and integrals of the input matrix must be calculated. For the
derivative and integral across the given interval, the input matrix is converted
to

.. math::
   :label: mnp-cubic-dx

   \frac{d}{d x} \bX =
   \begin{bmatrix} 0 & \Delta_{i}^{-1} & 2 (\Delta_{i}^{-1}) \, \tx & 3 (\Delta_{i}^{-1}) \, \tx^2 \end{bmatrix}

.. math::
   :label: mnp-cubic-int-01

   \int_{x_{i}}^{x_{i+1}} dx \, \bX =
   \begin{bmatrix} \Delta_{i} & \frac{1}{2} \Delta_{i} & \frac{1}{3} \Delta_{i} & \frac{1}{4} \Delta_{i} \end{bmatrix}

.. math::
   :label: mnp-cubic-dx-int-01

   \int_{x_{i}}^{x_{i+1}} dx \, \left( \frac{d}{d x} \bX \right) =
   \begin{bmatrix} 0 & 1 & 1 & 1 \end{bmatrix}

while the input matrix for the indefinite integrals would be represented as

.. math::
   :label: mnp-cubic-int-02

   \int dx \, \bX =
   \begin{bmatrix} x & \frac{1}{2} \Delta_{i}^{-1} \, x \, (x - 2 \, x_{i}) & \frac{1}{3} (x - x_{i}) \, \tx^2 & \frac{1}{4} (x - x_{i}) \, \tx^3 \end{bmatrix}

.. math::
   :label: mnp-cubic-dx-int-02

   \int dx \, \left( \frac{d}{d x} \bX \right) =
   \begin{bmatrix} 0 & \Delta_{i}^{-1} \, x & \Delta_{i}^{-2} \, x \, (x - 2 \, x_{i}) & \tx^3 \end{bmatrix}

for more generalized cases. These formulations will be used later in this chapter
when calculating the curve fit residual.


Bicubic Hermite Splines
-----------------------

The cubic Hermite splines have been formulated to work in one spatial dimension
:math:`x`. Bicubic interpolation :cite:`deboor1962` is used to project these
cubic Hermite splines onto a two-dimensional surface :math:`(x, y)`. In this
process, the :math:`S_i(x) = \bX \, \bB \, \bu` one-dimensional spline equation is
used to create a spline on the intervals :math:`[x_{i}, x_{i+1}]` and
:math:`[y_{j}, y_{j+1}]`. Taking the control matrix :math:`\bu` from the
one-dimensional case for both the :math:`x` and :math:`y` directions, a new
control matrix :math:`\bU = \bu_x \, \bu_y^T` can be formulated as

.. math::

   \bU =
   \begin{bmatrix}
   \psi_{i, j} & \psi_{i, j+1} & \Delta_{j} \, \partial_{y} \psi_{i, j} & \Delta_{j} \, \partial_{y} \psi_{i, j+1} \\
   \psi_{i+1, j} & \psi_{i+1, j+1} & \Delta_{j} \, \partial_{y} \psi_{i+1, j} & \Delta_{j} \, \partial_{y} \psi_{i+1, j+1} \\
   \Delta_{i} \, \partial_{x} \psi_{i, j} & \Delta_{i} \, \partial_{x} \psi_{i, j+1} & \Delta_{j} \Delta_{i} \, \partial_{yx} \psi_{i, j} & \Delta_{j} \Delta_{i} \, \partial_{yx} \psi_{i, j+1} \\
   \Delta_{i} \, \partial_{x} \psi_{i+1, j} & \Delta_{i} \, \partial_{x}  \psi_{i+1, j+1} & \Delta_{j} \Delta_{i} \, \partial_{yx} \psi_{i+1, j} & \Delta_{j} \Delta_{i} \, \partial_{yx} \psi_{i+1, j+1}
   \end{bmatrix}

where :math:`\psi_{i,j} = \psi(x_{i}, y_{j})`,
:math:`\Delta_{i} = x_{i+1} - x_{i}`, and :math:`\Delta_{j} = y_{j+1} - y_{j}`. It
should also be noted that :math:`\partial_{x}` and :math:`\partial_{y}` are first
derivatives with respect to :math:`x` and :math:`y`, and :math:`\partial_{yx}` is
the second derivative, all of which have been calculated using the formulas in
the :ref:`finite difference section <sec-mnp-derivative>`. Using the basis matrix
:math:`\bB` from :eq:`mnp-cubic-basis`, the two-dimensional spline can be
constructed as :math:`S_{i,j}(x,y) = \bX \, \bB \, \bU \, \bB^T \, \bY`. The
:math:`x` input matrix combines the rows :math:`[1, \; \tx, \; \tx^2, \; \tx^3]`
while the :math:`y` input matrix is a series of columns represented as
:math:`[1, \; \ty, \; \ty^2, \; \ty^3]^T`. For clarity, :math:`\tx` and
:math:`\ty` are normalized from the spline knots as

.. math::
   :label: mnp-normalize-xy

   \tx = \frac{x - x_{i}}{x_{i+1} - x_{i}} \qquand
   \ty = \frac{y - y_{j}}{y_{j+1} - y_{j}}

for the cell interval :math:`[x_{i}, x_{i+1}]` and :math:`[y_{j}, y_{j+1}]`. When
preparing the bicubic splines for application in the neutron transport equation,
the derivatives and integrals of the input matrix :math:`\bX` are the same as
those used for the cubic Hermite splines, namely :eq:`mnp-cubic-dx`,
:eq:`mnp-cubic-int-01`, :eq:`mnp-cubic-dx-int-01`, :eq:`mnp-cubic-int-02`, and
:eq:`mnp-cubic-dx-int-02`. The derivatives and integrals for the :math:`\bY`
input matrix are formulated in the same manner.


Quintic Hermite Splines
-----------------------

The quintic Hermite spline :math:`S_{i}(x)` can be constructed from the general
fifth order polynomial

.. math::

   S_{i}(x) = a_{i} + b_{i} (x - x_{i}) + c_{i} (x - x_{i})^2 + d_{i} (x - x_{i})^3
   + e_{i} (x - x_{i})^4 + f_{i} (x - x_{i})^5

where :math:`a_{i}`, :math:`b_{i}`, :math:`c_{i}`, :math:`d_{i}`, :math:`e_{i}`,
and :math:`f_{i}` are the coefficients to solve for with
:math:`i \in \{1, I-1 \}` number of splines :cite:`roy2007`. The interval
:math:`[x_{i}, x_{i+1}]` is chosen for the specific spline :math:`S_{i}(x)` where
:math:`x_{i}` and :math:`x_{i+1}` are the knot points. The constraints for the
fifth order polynomial are

.. math::

   \begin{aligned}
   S_{i}(x_{i}) &= \psi_{i} \\
   S_{i}(x_{i+1}) &= \psi_{i+1} \\
   \partial_{x} S_{i}(x_{i}) &= \partial_{x} \psi_{i} \\
   \partial_{x} S_{i}(x_{i+1}) &= \partial_{x} \psi_{i+1} \\
   \partial_{xx} S_{i}(x_{i}) &= \partial_{xx} \psi_{i} \\
   \partial_{xx} S_{i}(x_{i+1}) &= \partial_{xx} \psi_{i+1}
   \end{aligned}

where :math:`\partial_{x}` and :math:`\partial_{xx}` are the first and second
derivatives of the spline knots. It can be converted into the form

.. math::
   :label: mnp-quintic-eq

   S_{i}(x) = \psi_{i} \, h_{0,0}(x) + \psi_{i+1} \, h_{1,0}(x)
   + \partial_{x} (\psi_{i}) \, h_{0,1}(x) + \partial_{x} (\psi_{i+1}) \, h_{1,1}(x)
   + \partial_{xx} (\psi_{i}) \, h_{0,2}(x) + \partial_{xx} (\psi_{i+1}) \, h_{1,2}(x),

where the basis functions are :math:`h_{0,0}(x)`, :math:`h_{1,0}(x)`,
:math:`h_{0,1}(x)`, :math:`h_{1,1}(x)`, :math:`h_{0,2}(x)`, and
:math:`h_{1,2}(x)`. The :math:`x` values are normalized in the interval
:math:`[x_{i}, x_{i+1}]` according to :eq:`mnp-interp-tx` and represented as
:math:`\tx`. Representing the interval width as
:math:`\Delta_{i} = x_{i+1} - x_{i}`, the basis functions are expressed as

.. math::

   \begin{aligned}
   h_{0,0}(x) &= -6 \tx^5 + 15 \tx^4 - 10 \tx^3 + 1 \\
   h_{1,0}(x) &= 6 \tx^5 - 15 \tx^4 + 10 \tx^3 \\
   h_{0,1}(x) &= \Delta_{i} (-3 \tx^5 + 8 \tx^4 - 6 \tx^3 + \tx) \\
   h_{1,1}(x) &= \Delta_{i} ( -3 \tx^5 + 7 \tx^4 - 4 \tx^3 ) \\
   h_{0,2}(x) &= \Delta_{i}^{2} \left( -\frac{1}{2} \tx^5 + \frac{3}{2} \tx^4 - \frac{3}{2} \tx^3 + \frac{1}{2} \tx^2 \right) \\
   h_{1,2}(x) &= \Delta_{i}^{2} \left( \frac{1}{2} \tx^5 - \tx^4 + \frac{1}{2} \tx^3 \right).
   \end{aligned}

Quintic Hermite splines can be constructed using :eq:`mnp-quintic-eq` and the
basis functions.

While the quintic Hermite splines can be constructed using these equations, it is
more beneficial when moving to higher dimensions to convert the equation to matrix
form. The matrix form equation is :math:`S_{i}(x) = \bX \bB \bu` where
:math:`\bX` is the input matrix, :math:`\bB` is the basis matrix, and :math:`\bu`
is the control vector. The input matrix is a series of rows with each :math:`x`
input being :math:`[1, \; \tx, \; \tx^2, \; \tx^3, \; \tx^4, \; \tx^5]`. The
control vector is
:math:`[\psi_{i}, \; \psi_{i+1}, \; \Delta_{i} \, \partial_x \psi_{i}, \; \Delta_{i} \, \partial_x \psi_{i+1}, \; \Delta_{i}^2 \, \partial_{xx} \psi_{i}, \; \Delta_{i}^2 \, \partial_{xx} \psi_{i+1}]^T`
and the basis matrix is

.. math::
   :label: mnp-quintic-basis

   \bB =
   \begin{bmatrix}
   1 & 0 & 0 & 0 & 0 & 0 \\
   0 & 0 & 1 & 0 & 0 & 0 \\
   0 & 0 & 0 & 0 & 0.5 & 0 \\
   -10 & 10 & -6 & -4 & -1.5 & 0.5 \\
   15 & -15 & 8 & 7 & 1.5 & -1 \\
   -6 & 6 & -3 & -3 & -0.5 & 0.5
   \end{bmatrix},

which is combined for calculating the quintic Hermite spline :math:`S_{i}(x)`.

For the method of nearby problems, the derivative and integrals of the input
matrix are required for the residual calculation. With the quintic Hermite
splines, these are

.. math::
   :label: mnp-quintic-dx

   \frac{d}{d x} \bX =
   \begin{bmatrix} 0 & \Delta_{i}^{-1} & 2 (\Delta_{i}^{-1}) \, \tx & 3 (\Delta_{i}^{-1}) \, \tx^2 & 4 (\Delta_{i}^{-1}) \, \tx^3 & 5 (\Delta_{i}^{-1}) \, \tx^4 \end{bmatrix}

.. math::
   :label: mnp-quintic-int-01

   \int_{x_{i}}^{x_{i+1}} dx \, \bX =
   \begin{bmatrix} \Delta_{i} & \frac{1}{2} \Delta_{i} & \frac{1}{3} \Delta_{i} & \frac{1}{4} \Delta_{i} & \frac{1}{5} \Delta_{i} & \frac{1}{6} \Delta_{i} \end{bmatrix}

.. math::
   :label: mnp-quintic-dx-int-01

   \int_{x_{i}}^{x_{i+1}} dx \, \left( \frac{d}{d x} \bX \right) =
   \begin{bmatrix} 0 & 1 & 1 & 1 & 1 & 1 \end{bmatrix}

when integrating over the interval. The indefinite integrals are

.. math::
   :label: mnp-quintic-int-02

   \int dx \, \bX =
   \begin{bmatrix} x & \frac{1}{2} \Delta_{i}^{-1} \, x \, (x - 2 \, x_{i}) & \frac{1}{3} (x - x_{i}) \, \tx^2 & \frac{1}{4} (x - x_{i}) \, \tx^3 & \frac{1}{5} (x - x_{i}) \, \tx^4 & \frac{1}{6} (x - x_{i}) \, \tx^5 \end{bmatrix}

and

.. math::
   :label: mnp-quintic-dx-int-02

   \int dx \, \left( \frac{d}{d x} \bX \right) =
   \begin{bmatrix} 0 & \Delta_{i}^{-1} \, x & \Delta_{i}^{-2} \, x \, (x - 2 \, x_{i-1/2}) & \tx^3 & \tx^4 & \tx^5 \end{bmatrix}

for the generalized cases.


Biquintic Hermite Splines
-------------------------

Biquintic interpolation is used to project the quintic Hermite splines formulated
in the previous section onto a two-dimensional surface. This is the expansion of
the bicubic interpolation in :cite:`deboor1962`. For this process, the spline
equation :math:`S_i(x) = \bX \bB \bu` is used to create a spline on the intervals
:math:`[x_{i}, x_{i+1}]` and :math:`[y_{j}, y_{j+1}]`. The control matrix
:math:`\bU = \bu_x \bu_y^T` can be represented as

.. math::

   \bU =
   \begin{bmatrix}
   \bu_{0, 0} & \bu_{0, 1} & \bu_{0, 2} \\
   \bu_{1, 0} & \bu_{1, 1} & \bu_{1, 2} \\
   \bu_{2, 0} & \bu_{2, 1} & \bu_{2, 2}
   \end{bmatrix}

where each sub-matrix is represented as

.. math::

   \begin{aligned}
   \bu_{0,0} &= \begin{bmatrix}
                   \psi_{i, j} & \psi_{i, j+1} \\
                   \psi_{i+1, j} & \psi_{i+1, j+1}
               \end{bmatrix}, &
   \bu_{0,1} &= \begin{bmatrix}
                   \Delta_{j} \, \partial_{y} \psi_{i, j} & \Delta_{j} \, \partial_{y} \psi_{i, j+1} \\
                   \Delta_{j} \, \partial_{y} \psi_{i+1, j} & \Delta_{j} \, \partial_{y} \psi_{i+1, j+1}
               \end{bmatrix}, \\[5pt]
   \bu_{0,2} &= \begin{bmatrix}
                   \Delta_{j}^2 \, \partial_{yy} \psi_{i, j} & \Delta_{j}^2 \, \partial_{yy} \psi_{i, j+1} \\
                   \Delta_{j}^2 \, \partial_{yy} \psi_{i+1, j} & \Delta_{j}^2 \, \partial_{yy} \psi_{i+1, j+1}
               \end{bmatrix}, &
   \bu_{1,0} &= \begin{bmatrix}
                   \Delta_{i} \, \partial_{x} \psi_{i, j} & \Delta_{i} \, \partial_{x} \psi_{i, j+1} \\
                   \Delta_{i} \, \partial_{x} \psi_{i+1, j} & \Delta_{i} \, \partial_{x} \psi_{i+1, j+1}
               \end{bmatrix}, \\[5pt]
   \bu_{1,1} &= \begin{bmatrix}
                   \Delta_{j} \Delta_{i} \, \partial_{yx} \psi_{i, j} & \Delta_{j} \Delta_{i} \, \partial_{yx} \psi_{i, j+1} \\
                   \Delta_{j} \Delta_{i} \, \partial_{yx} \psi_{i+1, j} & \Delta_{j} \Delta_{i} \, \partial_{yx} \psi_{i+1, j+1}
               \end{bmatrix}, &
   \bu_{1,2} &= \begin{bmatrix}
                   \Delta_{j}^2 \Delta_{i} \, \partial_{yyx} \psi_{i, j} & \Delta_{j}^2 \Delta_{i} \, \partial_{yyx} \psi_{i, j+1} \\
                   \Delta_{j}^2 \Delta_{i} \, \partial_{yyx} \psi_{i+1, j} & \Delta_{j}^2 \Delta_{i} \, \partial_{yyx} \psi_{i+1, j+1}
               \end{bmatrix}, \\[5pt]
   \bu_{2,0} &= \begin{bmatrix}
                   \Delta_{i}^2 \, \partial_{xx} \psi_{i, j} & \Delta_{i}^2 \, \partial_{xx} \psi_{i, j+1} \\
                   \Delta_{i}^2 \, \partial_{xx} \psi_{i+1, j} & \Delta_{i}^2 \, \partial_{xx} \psi_{i+1, j+1}
               \end{bmatrix}, &
   \bu_{2,1} &= \begin{bmatrix}
                   \Delta_{j} \Delta_{i}^2 \, \partial_{yxx} \psi_{i, j} & \Delta_{j} \Delta_{i}^2 \, \partial_{yxx} \psi_{i, j+1} \\
                   \Delta_{j} \Delta_{i}^2 \, \partial_{yxx} \psi_{i+1, j} & \Delta_{j} \Delta_{i}^2 \, \partial_{yxx} \psi_{i+1, j+1}
               \end{bmatrix}, \\[5pt]
   \bu_{2,2} &= \begin{bmatrix}
                   \Delta_{j}^2 \Delta_{i}^2 \, \partial_{yyxx} \psi_{i, j} & \Delta_{j}^2 \Delta_{i}^2 \, \partial_{yyxx} \psi_{i, j+1} \\
                   \Delta_{j}^2 \Delta_{i}^2 \, \partial_{yyxx} \psi_{i+1, j} & \Delta_{j}^2 \Delta_{i}^2 \, \partial_{yyxx} \psi_{i+1, j+1}
               \end{bmatrix},
   \end{aligned}

where :math:`\psi_{i,j} = \psi(x_{i}, y_{j})`,
:math:`\Delta_{i} = x_{i+1} - x_{i}`, and :math:`\Delta_{j} = y_{j+1} - y_{j}`
with the numerical derivatives calculated in the :ref:`finite difference section
<sec-mnp-derivative>`. Using the control matrix :math:`\bU` and the basis matrix
in :eq:`mnp-quintic-basis`, the two-dimensional spline formula is
:math:`S_{i,j}(x,y) = \bX \bB \bU \bB^T \bY`. The :math:`x` input matrix is made up
of rows :math:`[1, \; \tx, \; \tx^2, \; \tx^3 \; \tx^4, \; \tx^5]` while the
:math:`y` input matrix is a series of columns represented as
:math:`[1, \; \ty, \; \ty^2, \; \ty^3 \; \ty^4, \; \ty^5]^T` with :math:`\tx` and
:math:`\ty` being the normalized spline points in :eq:`mnp-normalize-xy`. The
derivatives and integrals of the input matrices are the same as the
one-dimensional quintic splines and employ :eq:`mnp-quintic-dx`,
:eq:`mnp-quintic-int-01`, :eq:`mnp-quintic-dx-int-01`, :eq:`mnp-quintic-int-02`,
and :eq:`mnp-quintic-dx-int-02`. Altering the :math:`\bY` input matrix for
derivatives and integrals is performed in the same manner.
