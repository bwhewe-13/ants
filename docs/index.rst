A Neutron Transport Solution (ANTS)
===========================================

A Neutron Transport Solution (ANTS) solves the neutron transport equation with
the discrete ordinates (:math:`S_N`) method. It is written in Cython and
parallelized over angles and energy groups with OpenMP. ANTS solves fixed
source, criticality, and time-dependent problems in one dimensional slabs and
spheres and two dimensional slabs, and fixed source problems in three
dimensional slabs, with isotropic or anisotropic scattering. The iterations can
be accelerated with dynamic mode decomposition and with the collision-based
hybrid method, and ray effects can be reduced with artificial scattering.

.. code-block:: bash

   python -m pip install ants-sn

The package is imported as ``ants``. To get started, see
:doc:`running-ants/getting-started` and the :doc:`running-ants/examples`.

* Source code: https://github.com/bwhewe-13/ants
* Package: https://pypi.org/project/ants-sn/
* Changes in each version: `changelog <https://github.com/bwhewe-13/ants/blob/main/CHANGELOG.md>`_


Citing ANTS
-----------

If you use ANTS in your work, please cite it with the DOI
`10.5281/zenodo.24066032 <https://doi.org/10.5281/zenodo.24066032>`_, which
always points to the latest version. The methods are described in
:cite:`whewell2023,whewell2025,whewell2024`.


Contents
--------

.. toctree::
   :maxdepth: 1

   Running ANTS <running-ants/index>

   Neutron Transport Equation <neutron-transport/index>

   Collision-Based Hybrid Method <hybrid-method/index>

   Method of Nearby Problems <nearby-problems/index>

   References <references>
