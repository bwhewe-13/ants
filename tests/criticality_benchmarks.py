"""Criticality Benchmark Setup.

Taken from ``Benchmark Test Set for Criticality Code Verification`` by
Sood, Forster, and Parsons(1999).
"""

import numpy as np

import ants
from ants.datatypes import GeometryData, MaterialData


def PUa_1_0(cells_x, bc_x):
    materials = MaterialData(
        total=np.array([[0.32640]]),
        scatter=np.array([[[0.225216]]]),
        fission=np.array([[[3.24 * 0.0816]]]),
    )
    length = 1.853722 if np.sum(bc_x) == 1 else 1.853722 * 2
    medium_map = np.zeros((cells_x), dtype=np.int32)
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=np.repeat(length / cells_x, cells_x),
        bc_x=bc_x,
        geometry=1,
    )
    return materials, geometry


def PUb_1_0(cells_x, bc_x, geometry_type):
    materials = MaterialData(
        total=np.array([[0.32640]]),
        scatter=np.array([[[0.225216]]]),
        fission=np.array([[[2.84 * 0.0816]]]),
    )
    if geometry_type == 1:
        length = 2.256751 if np.sum(bc_x) == 1 else 2.256751 * 2
    elif geometry_type == 2:
        length = 6.082547
    medium_map = np.zeros((cells_x), dtype=np.int32)
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=np.repeat(length / cells_x, cells_x),
        bc_x=bc_x,
        geometry=geometry_type,
    )
    return materials, geometry


def PUa_H20_1_0(cells_x, nonsymmetric=True):
    materials = MaterialData(
        total=np.array([[0.32640], [0.32640]]),
        scatter=np.array([[[0.225216]], [[0.293760]]]),
        fission=np.array([[[3.24 * 0.0816]], [[0.0]]]),
    )
    if nonsymmetric:
        length = 1.478401 * 2 + 3.063725
        edges_x = np.linspace(0, length, cells_x + 1)
        layout = [
            [0, "fuel", "0 - 2.956802"],
            [1, "moderator", "2.956802 - 6.020527"],
        ]
    else:
        length = np.sum([1.531863, 1.317831 * 2, 1.531863])
        edges_x = np.linspace(0, length, cells_x + 1)
        layout = [
            [0, "fuel", "1.531863 - 4.167525"],
            [1, "moderator", "0 - 1.531863, 4.167525 - 5.699388"],
        ]
    medium_map = ants.spatial1d(layout, edges_x)
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=np.repeat(length / cells_x, cells_x),
        bc_x=[0, 0],
        geometry=1,
    )
    return materials, geometry


def Ua_1_0(cells_x, bc_x, geometry_type=1):
    materials = MaterialData(
        total=np.array([[0.32640]]),
        scatter=np.array([[[0.248064]]]),
        fission=np.array([[[2.70 * 0.065280]]]),
    )
    if geometry_type == 1:
        length = 2.872934 if np.sum(bc_x) == 1 else 2.872934 * 2
    elif geometry_type == 2:
        length = 7.428998
    medium_map = np.zeros((cells_x), dtype=np.int32)
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=np.repeat(length / cells_x, cells_x),
        bc_x=bc_x,
        geometry=geometry_type,
    )
    return materials, geometry


def UD2O_1_0(cells_x, bc_x, geometry_type=1):
    materials = MaterialData(
        total=np.array([[0.54628]]),
        scatter=np.array([[[0.464338]]]),
        fission=np.array([[[1.70 * 0.054628]]]),
    )
    if geometry_type == 1:
        length = 10.371065 if np.sum(bc_x) == 1 else 10.371065 * 2
    elif geometry_type == 2:
        length = 22.017156
    medium_map = np.zeros((cells_x), dtype=np.int32)
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=np.repeat(length / cells_x, cells_x),
        bc_x=bc_x,
        geometry=geometry_type,
    )
    return materials, geometry


def PU_2_0(cells_x, bc_x, geometry_type):
    """PU-2-0 (Problems 44-46): Two-Group Pu-239, Isotropic Scattering.

    Group 0 = slow (group 1 in report), Group 1 = fast (group 2 in report).
    Tables 30-31 in Sood, Forster, Parsons (1999).
    """
    chi = np.array([[0.425], [0.575]])
    nu = np.array([[2.93, 3.10]])
    sigma_f = np.array([[0.08544, 0.0936]])
    materials = MaterialData(
        total=np.array([[0.3360, 0.2208]]),
        scatter=np.array([[[0.23616, 0.0432], [0.0, 0.0792]]]),
        fission=np.array([chi @ (nu * sigma_f)]),
    )
    if geometry_type == 1:
        length = 1.795602 if np.sum(bc_x) == 1 else 1.795602 * 2
    elif geometry_type == 2:
        length = 5.231567
    medium_map = np.zeros((cells_x), dtype=np.int32)
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=np.repeat(length / cells_x, cells_x),
        bc_x=bc_x,
        geometry=geometry_type,
    )
    return materials, geometry


def U_2_0(cells_x, bc_x, geometry_type):
    """U-2-0 (Problems 47-49): Two-Group U-235, Isotropic Scattering.

    Group 0 = slow (group 1 in report), Group 1 = fast (group 2 in report).
    Tables 33-34 in Sood, Forster, Parsons (1999).
    """
    chi = np.array([[0.425], [0.575]])
    nu = np.array([[2.50, 2.70]])
    sigma_f = np.array([[0.06912, 0.06192]])
    materials = MaterialData(
        total=np.array([[0.3456, 0.2160]]),
        scatter=np.array([[[0.26304, 0.0720], [0.0, 0.078240]]]),
        fission=np.array([chi @ (nu * sigma_f)]),
    )
    if geometry_type == 1:
        length = 3.006375 if np.sum(bc_x) == 1 else 3.006375 * 2
    elif geometry_type == 2:
        length = 7.909444
    medium_map = np.zeros((cells_x), dtype=np.int32)
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=np.repeat(length / cells_x, cells_x),
        bc_x=bc_x,
        geometry=geometry_type,
    )
    return materials, geometry


def UAL_2_0(cells_x, bc_x, geometry_type):
    """UAL-2-0 (Problems 50-52): Two-Group U-Al Assembly, Isotropic Scattering.

    Group 0 = slow (group 1 in report), Group 1 = fast (group 2 in report).
    Fission only in slow group; all fission neutrons born in fast group.
    Tables 36-37 in Sood, Forster, Parsons (1999).
    """
    chi = np.array([[0.0], [1.0]])
    nu = np.array([[2.83, 0.0]])
    sigma_f = np.array([[0.06070636042, 0.0]])
    materials = MaterialData(
        total=np.array([[1.27698, 0.26817]]),
        scatter=np.array([[[1.21313, 0.020432], [0.0, 0.247516]]]),
        fission=np.array([chi @ (nu * sigma_f)]),
    )
    if geometry_type == 1:
        length = 7.830630 if np.sum(bc_x) == 1 else 7.830630 * 2
    elif geometry_type == 2:
        length = 17.66738
    medium_map = np.zeros((cells_x), dtype=np.int32)
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=np.repeat(length / cells_x, cells_x),
        bc_x=bc_x,
        geometry=geometry_type,
    )
    return materials, geometry


def URRa_2_0(cells_x, bc_x, geometry_type):
    """URRa-2-0 (Problems 53-55): Two-Group Research Reactor (a), Isotropic Scattering.

    Group 0 = slow (group 1 in report), Group 1 = fast (group 2 in report).
    Fission neutrons born only in fast group.
    Tables 39-40 in Sood, Forster, Parsons (1999).
    """
    chi = np.array([[0.0], [1.0]])
    nu = np.array([[2.50, 2.50]])
    sigma_f = np.array([[0.050632, 0.0010484]])
    materials = MaterialData(
        total=np.array([[2.52025, 0.65696]]),
        scatter=np.array([[[2.44383, 0.029227], [0.0, 0.62568]]]),
        fission=np.array([chi @ (nu * sigma_f)]),
    )
    if geometry_type == 1:
        length = 7.566853 if np.sum(bc_x) == 1 else 7.566853 * 2
    elif geometry_type == 2:
        length = 16.049836
    medium_map = np.zeros((cells_x), dtype=np.int32)
    geometry = GeometryData(
        medium_map=medium_map,
        delta_x=np.repeat(length / cells_x, cells_x),
        bc_x=bc_x,
        geometry=geometry_type,
    )
    return materials, geometry
