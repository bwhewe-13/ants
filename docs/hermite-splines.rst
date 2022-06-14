Basis Functions for Hermite Splines
===============================================

Hermite Splines are used for approximating the angular flux with the Method
of Nearby Problms (MNB).


Cubic Hermite Splines
---------------------

If we were to construct a cubic Hermite spline :math:`S_k` in the interval :math:`[t_{k}, t_{k+1}]`, we can assume the form 

.. math::
   :label: h3-initial

   \begin{equation}
       S_k(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3.
   \end{equation}

Using the constraints 

.. math::

   \begin{equation} \begin{split}
       S_k(t_{k}) = y_{k} \qquad S_k(t_{k+1}) = y_{k+1} \\
       S_k'(t_{k}) = y'_{k} \qquad S_k'(t_{k+1}) = y'_{k+1}
   \end{split} \end{equation}

the system of equations that needs to be solved is

.. math::

   \begin{equation} \begin{split}
       & \phi_0 = a_0 \\
       & \psi_0 = a_1 \\
       & \phi_1 = a_0 + a_1 + a_2 + a_3 \\
       & \psi_1 = a_1 + 2 a_2 + 3 a_3.
   \end{split} \end{equation}

Solving and converting this to the basis function form 

.. math::

   \begin{equation}
       S_k(x) = y_{k} \phi_0(x) + y_{k+1} \phi_1(x) + y'_{k} \psi_0(x) + y'_{k+1} \psi_1(x), 
   \end{equation}

where the basis functions are 

.. math::

   \begin{equation} \begin{split}
       & \phi_0(x) = 2 t^3 - 3 t^2 + 1 \\
       & \phi_1(x) = -2 t^3 + 3 t^2 \\
       & \psi_0(x) = \Delta_k \left( t^3 - 2 t^2 + t \right) \\
       & \psi_1(x) = \Delta_k \left( t^3 - t^2 \right)
   \end{split} \end{equation}

and

.. math::
   :label: sym-key

   \begin{equation} 
       \Delta_k = t_{k+1} - t_{k}, \qquad t = \frac{x - t_{k}}{t_{k+1} - t_{k}}.
   \end{equation} 

If the derivatives at the points :math:`t_{k}` and :math:`t_{k+1}`, they can be calculated via the central difference approximation 

.. math::
   :label: yp1

   \begin{equation} 
       y'_{k} = \frac{y_{k+1} - y_{k-1}}{t_{k+1} - t_{k-1}} \qquad 1 \leq k \leq n-1
   \end{equation} 

and the backward and forward difference approximation 

.. math::
   :label: yp2

   \begin{equation} \begin{split} 
       y'_0 = \frac{y_1 - y_0}{t_1 - t_0} \qquad
       y'_n = \frac{y_n - y_{n-1}}{t_n - t_{n-1}} 
   \end{split} \end{equation} 

at the boundary points :math:`k = 0` and :math:`k = n`.


Quintic Hermite Splines
-----------------------

If we were to construct a quintic Hermite spline :math:`S_k` in the interval :math:`[t_{k}, t_{k+1}]` with :math:`t_{k} \leq x t_{k+1}` and :math:`0 \leq k \leq n`, we can assume the form 

.. math::

   \begin{equation}
       S_k(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3 + a_4 x^4 + a_5 x^5.
   \end{equation} 

The constraints would be 

.. math:: 

   \begin{equation} \begin{split}
       S_k(t_{k}) = y_{k} \qquad S_k(t_{k+1}) = y_{k+1} \\
       S_k'(t_{k}) = y'_{k} \qquad S_k'(t_{k+1}) = y'_{k+1} \\
       S_k''(t_{k}) = y''_{k} \qquad S_k''(t_{k+1}) = y''_{k+1} \\
   \end{split} \end{equation} 

and the system of equations that needs to be solved is 

.. math:: 

   \begin{equation} \begin{split}
       & \phi_0 = a_0 \\
       & \psi_0 = a_1 \\
       & \theta_0 = a_2 \\
       & \phi_1 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5 \\
       & \psi_1 = a_1 + 2 a_2 + 3 a_3 + 4 a_4 + 5 a_5 \\
       & \theta_1 = 2 a_2 + 6 a_3 + 12 a_4 + 20 a_5.
   \end{split} \end{equation}

Solving and converting this to a basis function form, we have 

.. math::

   \begin{equation}
        S_k(x) = y_{k} \phi_0(x) + y_{k+1} \phi_1(x) + y'_{k} \psi_0(x) + y'_{k+1} \psi_1(x) + y''_{k} \theta_0(x) + y''_{k+1} \theta_1(x), 
   \end{equation}

where the basis functions are 

.. math::

   \begin{equation} \begin{split}
       & \phi_0(x) = -6 t^5 + 15 t^4 - 10 t^3 + 1 \\
       & \phi_1(x) = 6 t^5 - 15 t^4 + 10 t^3 \\
       & \psi_0(x) = \Delta_k \left( -3 t^5 + 8 t^4 - 6 t^3 + t \right) \\
       & \psi_1(x) = \Delta_k \left( -3 t^5 + 7 t^4 - 4 t^3 \right) \\
       & \theta_0(x) = \Delta_k^2 \left( -\frac{1}{2} t^5 + \frac{3}{2} t^4 - \frac{3}{2} t^3 + \frac{1}{2} t^2 \right) \\
       & \theta_1(x) = \Delta_k^2 \left( \frac{1}{2} t^5 - t^4 + \frac{1}{2} t^3 \right), 
   \end{split} \end{equation}

where :math:`t` and :math:`\Delta_k` have been explained in :eq:`sym-key`.
The first derivative is calculated in :eq:`yp1` and :eq:`yp2` at the points :math:`t_{k}` and :math:`t_{k+1}`.
To calculate the second derivative at these point, the central difference approximation is used via 

.. math::

   \begin{equation} \label{eq:ypp1}
       y''_{k} = \frac{y_{k+1} - 2 y_{k} + y_{k-1}}{(t_{k+1} - t_{k}) (t_{k} - t_{k-1}) } \qquad 1 \leq k \leq n-1
   \end{equation} 

and the backward and forward difference approximation 

.. math::
   :label: ypp2
   
   \begin{equation} \begin{split} 
       y''_0 = \frac{y_0 - 2 y_1 + y_2}{(t_2 - t_1)(t_1 - t_0)} \qquad
       y''_n = \frac{y_n - 2 y_{n-1} + y_{n-2}}{(t_n - t_{n-1}) (t_{n-1} - t_{n-2})} 
   \end{split} \end{equation}

at the boundary points :math:`k = 0` and :math:`k = n`.


