########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Reader for HDF5 multigroup cross section libraries
#
########################################################################

"""Read microscopic cross sections from an HDF5 multigroup library.

The library is a single HDF5 file with the layout below (G groups, L
Legendre moments). Cross sections are microscopic, in barns.

::

    library.h5
    ├── attrs
    │   ├── n_groups        int, G
    │   ├── n_legendre      int, L (optional)
    │   └── library_type    "microscopic" (optional)
    ├── group_boundaries    (G + 1,) energy bounds in eV, descending
    └── <nuclide>           one group per nuclide, e.g. "U-235", or
        │                   "U-235_293.6K" when stored at several
        │                   temperatures (or set attrs["temperature"])
        ├── xs_total        (G,)
        ├── xs_scatter      (G, G, L) or (G, G), indexed [in, out, l]
        ├── xs_fission      (G,)            fissile nuclides only
        ├── nu              (G,)            fissile nuclides only
        ├── chi_matrix      (G, G) [in, out], or
        └── chi             (G,) fission spectrum

Group 0 is the highest energy group. Nuclide names follow ``"Pu-239"``,
``"H-1"``, ``"Am-242m"``. Scattering is transposed here to the ANTS
``[out, in]`` convention. Only the P0 moment is used unless a Legendre
order is requested with ``legendre``, which keeps the moments on a last
axis, ``[out, in, l]``. The fission matrix is
``chi[in, out] * nu[in] * xs_fission[in]``.

h5py is an optional dependency (``pip install ants[library]``).
"""

import re
import warnings

import numpy as np

from ants.datatypes import MaterialData
from ants.main import energy_velocity
from ants.utils.hybrid import coarsen_materials, energy_coarse_index

# Library entries that are not nuclides
_RESERVED_KEYS = ("group_boundaries", "group_names", "compositions", "depletion")

# Temperature tag on a nuclide key, e.g. "U-235_293.6K"
_TEMP_KEY_RE = re.compile(r"_(\d+(?:\.\d+)?)K$")


def _import_h5py():
    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "Reading HDF5 libraries requires h5py: pip install ants[library]"
        ) from error
    return h5py


class Library:
    """An HDF5 multigroup cross section library.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the HDF5 library file (layout in the module docstring).

    Attributes
    ----------
    path : str
        Library file path.
    groups : int
        Number of energy groups.
    edges_g : ndarray, shape (groups + 1,)
        Energy group bounds in eV, descending.
    library_type : str
        ``"microscopic"`` or ``"macroscopic"``.
    """

    def __init__(self, path):
        h5py = _import_h5py()
        self.path = str(path)
        self._keys = {}
        self._cache = {}
        with h5py.File(self.path, "r") as f:
            self.groups = int(f.attrs["n_groups"])
            self.library_type = str(f.attrs.get("library_type", "microscopic"))
            self.edges_g = np.asarray(f["group_boundaries"][:], dtype=float)
            for key, item in f.items():
                if (key in _RESERVED_KEYS) or not isinstance(item, h5py.Group):
                    continue
                match = _TEMP_KEY_RE.search(key)
                if match:
                    name, temperature = key[: match.start()], float(match.group(1))
                else:
                    name = key
                    temperature = float(item.attrs.get("temperature", np.nan))
                self._keys.setdefault(name, []).append((temperature, key))
        for name in self._keys:
            self._keys[name].sort()

    def __repr__(self):
        return (
            f"Library({self.path!r}, groups={self.groups}, "
            f"nuclides={len(self._keys)})"
        )

    def __contains__(self, nuclide):
        return nuclide in self._keys

    def nuclides(self):
        """Return the sorted nuclide names in the library."""
        return sorted(self._keys)

    def temperatures(self, nuclide):
        """Return the available temperatures (K) for a nuclide."""
        return [temperature for temperature, _ in self._library_keys(nuclide)]

    def _library_keys(self, nuclide):
        if nuclide not in self._keys:
            raise KeyError(f"Nuclide {nuclide!r} is not in the library {self.path}")
        return self._keys[nuclide]

    def _nuclide_key(self, nuclide, temperature=None):
        # Nearest available temperature (lowest if not specified)
        options = self._library_keys(nuclide)
        if (temperature is None) or (len(options) == 1):
            return options[0][1]
        idx = np.nanargmin([abs(tt - temperature) for tt, _ in options])
        return options[idx][1]

    def microscopic(self, nuclide, temperature=None, legendre=0):
        """Load the microscopic cross sections of a single nuclide.

        Parameters
        ----------
        nuclide : str
            Nuclide name in the library, e.g. ``"U-235"``.
        temperature : float, optional
            Temperature (K); the nearest available temperature is used.
        legendre : int, optional
            Legendre order L of the scattering. 0 (default) returns the P0
            scatter matrix; L > 0 returns moments 0 to L on a last axis.

        Returns
        -------
        total : ndarray, shape (groups,)
            Total cross section (barns).
        scatter : ndarray, shape (groups, groups) or (groups, groups, L + 1)
            Scatter cross section, indexed ``[out, in]`` or
            ``[out, in, l]`` (barns).
        fission : ndarray, shape (groups, groups)
            Fission matrix chi * nu * sigma_f, indexed ``[out, in]`` (barns).
        """
        key = self._nuclide_key(nuclide, temperature)
        if key not in self._cache:
            self._cache[key] = self._read_nuclide(key)
        total, scatter, fission = self._cache[key]
        if legendre == 0:
            return total, scatter[..., 0], fission
        if legendre + 1 > scatter.shape[2]:
            # Nuclides without scattering data have no moments to check
            if np.any(scatter):
                raise ValueError(
                    f"Nuclide {key!r} has {scatter.shape[2]} Legendre moments, "
                    f"order {legendre} needs {legendre + 1}"
                )
            scatter = np.zeros((self.groups, self.groups, legendre + 1))
        return total, scatter[..., : legendre + 1], fission

    def _read_nuclide(self, key):
        h5py = _import_h5py()
        groups = self.groups
        with h5py.File(self.path, "r") as f:
            data = f[key]
            total = np.asarray(data["xs_total"][:], dtype=float)

            # Stored as [in, out, legendre], kept as [out, in, legendre]
            scatter = np.zeros((groups, groups, 1))
            if "xs_scatter" in data:
                xs_scatter = np.asarray(data["xs_scatter"][:], dtype=float)
                if xs_scatter.ndim == 2:
                    xs_scatter = xs_scatter[..., None]
                scatter = np.ascontiguousarray(xs_scatter.transpose(1, 0, 2))

            fission = np.zeros((groups, groups))
            if ("xs_fission" in data) and ("nu" in data):
                nu_fission = data["nu"][:] * data["xs_fission"][:]
                if "chi_matrix" in data:
                    # Stored as [in, out]
                    chi = np.asarray(data["chi_matrix"][:], dtype=float).T
                elif "chi" in data:
                    chi = np.repeat(data["chi"][:][:, None], groups, axis=1)
                else:
                    raise ValueError(f"Nuclide {key!r} has fission but no chi")
                fission = chi * nu_fission[None, :]

        return total, scatter, fission

    def macroscopic(self, composition, temperature=None, legendre=0):
        """Build macroscopic cross sections from number densities.

        Parameters
        ----------
        composition : dict[str, float]
            Nuclide names (library keys) and number densities
            (atoms/barn-cm).
        temperature : float, optional
            Temperature (K); the nearest available temperature is used.
        legendre : int, optional
            Legendre order L of the scattering (0 for P0 only).

        Returns
        -------
        xs_total : ndarray, shape (groups,)
        xs_scatter : ndarray, shape (groups, groups[, L + 1])
        xs_fission : ndarray, shape (groups, groups)
            Macroscopic cross sections (1/cm), indexed ``[out, in]``.
        """
        xs_total = np.zeros((self.groups,))
        if legendre == 0:
            xs_scatter = np.zeros((self.groups, self.groups))
        else:
            xs_scatter = np.zeros((self.groups, self.groups, legendre + 1))
        xs_fission = np.zeros((self.groups, self.groups))
        for nuclide, density in composition.items():
            total, scatter, fission = self.microscopic(nuclide, temperature, legendre)
            xs_total += density * total
            xs_scatter += density * scatter
            xs_fission += density * fission
        return xs_total, xs_scatter, xs_fission

    def materials(self, compositions, temperature=None, groups=None, legendre=0):
        """Build ``MaterialData`` for several materials.

        Parameters
        ----------
        compositions : list[dict[str, float]]
            Nuclide names and number densities (atoms/barn-cm) for each
            material.
        temperature : float, optional
            Temperature (K); the nearest available temperature is used.
        groups : int, optional
            Collapse to fewer energy groups. The collapse uses energy bin
            width weighting, not flux weighting; for a different energy
            structure build a library at that structure instead.
        legendre : int, optional
            Legendre order L of the scattering. 0 (default) gives P0
            scattering, shape ``(materials, groups, groups)``; L > 0 gives
            shape ``(materials, groups, groups, L + 1)``.

        Returns
        -------
        MaterialData
            Cross sections of shape ``(materials, groups[, groups])`` with
            the neutron velocity of each group.
        """
        xs_total = []
        xs_scatter = []
        xs_fission = []
        for composition in compositions:
            total, scatter, fission = self.macroscopic(
                composition, temperature, legendre
            )
            xs_total.append(total)
            xs_scatter.append(scatter)
            xs_fission.append(fission)
        xs_total = np.array(xs_total)
        xs_scatter = np.array(xs_scatter)
        xs_fission = np.array(xs_fission)

        edges_g = self.edges_g
        if (groups is not None) and (groups != self.groups):
            warnings.warn(
                f"Collapsing {self.groups} to {groups} groups with energy "
                "bin width weighting (not flux weighting)",
                stacklevel=2,
            )
            edges_gidx = energy_coarse_index(self.groups, groups)
            xs_total, xs_scatter, xs_fission = coarsen_materials(
                xs_total, xs_scatter, xs_fission, edges_g, edges_gidx
            )
            edges_g = edges_g[edges_gidx]

        velocity = energy_velocity(edges_g.shape[0] - 1, edges_g)
        return MaterialData(
            total=xs_total, scatter=xs_scatter, fission=xs_fission, velocity=velocity
        )
