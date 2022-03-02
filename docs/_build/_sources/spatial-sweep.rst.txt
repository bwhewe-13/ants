Spatial Sweeps
===============================================

The angular flux calculations for different geometries.

One Dimensional Slab
--------------------

Neutron Transport Equation

.. math::
   :label: 1d-nte

   \begin{equation}
   \frac{\mu}{\Delta x} \left( \Psi_{i + 1/2} - \Psi_{i - 1/2} \right) + \sigma_{t, i} \Psi_{i} = q_{i}
   \end{equation}

For the diamond difference method, using :eq:`1d-nte` with 

.. math::
   :label: 1d-dd

   \begin{equation}
   \Psi_{i} = \frac{1}{2} \left( \Psi_{i + 1/2} + \Psi_{i - 1/2} \right),
   \end{equation}

the angular flux calculation at the cell center is 

.. math::
   :label: 1d-dd-center

   \begin{equation}
   \Psi_{i} = \frac{q_{i} + \frac{2 \mu}{\Delta x} \Psi_{i - 1/2}}{\sigma_{t,i} + \frac{2 \mu}{\Delta x}}
   \end{equation}

The step method uses 

.. math::
   :label: 1d-st

   \begin{equation}
   \Psi_{i} = \begin{cases} \Psi_{i + 1/2} & \mu > 0 \\ 
                            \Psi_{i - 1/2} & \mu < 0
              \end{cases}
   \end{equation}

so the angular flux at the cell center is

.. math::
   :label: 1d-st-center

   \begin{equation}
   \Psi_{i} = \frac{q_{i} + \frac{\mu}{\Delta x} \Psi_{i - 1/2}}{\sigma_{t,i} + \frac{\mu}{\Delta x}}.
   \end{equation}

These are for the steady state one dimensional spatial sweep. 
To change these calculations for time-dependent use, :eq:`1d-nte` incorporates the time dependent term.
For the Backward Euler case, 

.. math::
   :label: 1d-be

   \begin{equation}
   \frac{1}{v} \frac{\partial \Psi}{\partial t} = \frac{1}{v \Delta t}\left( \Psi_{i}^{n+1} - \Psi_{i}^{n} \right),
   \end{equation}

where (n + 1) is the current time step.
Substituting into :eq:`1d-nte` results in

.. math::
   :label: 1d-nte-td

   \begin{equation}
   \frac{1}{v \Delta t}\left( \Psi_{i}^{n+1} - \Psi_{i}^{n} \right) + \frac{\mu}{\Delta x} \left( \Psi_{i + 1/2}^{n+1} - \Psi_{i - 1/2}^{n+1} \right) + \sigma_{t, i} \Psi_{i}^{n+1} = q_{i}
   \end{equation}


