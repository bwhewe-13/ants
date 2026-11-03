########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for the LDFE-SA quadrature option (method="ldfe") in angular_xy
# and angular_xyz. LDFE-SA sets come from Jarrell and Adams, M&C 2011.
#
########################################################################

import os

import numpy as np
import pytest

import ants
from tests.test_angular_ordering import _check_axis_ordering

LDFE_PATH = os.path.join(
    os.path.dirname(ants.__file__), "sources", "quadrature", "ldfe_sa.npz"
)

pytestmark = pytest.mark.skipif(
    not os.path.isfile(LDFE_PATH), reason="Local LDFE-SA quadrature is not available"
)

# LDFE-SA levels 1, 2, 3 correspond to these per-dimension angle counts.
LDFE_ANGLES = [4, 8, 16]


# ====================================================================
# angular_xy (method="ldfe") -- basic properties
# ====================================================================


@pytest.mark.parametrize("angles", LDFE_ANGLES)
def test_ldfe_xy_shape(angles):
    q = ants.angular_xy(angles, method="ldfe")
    assert q.angle_x.shape == (angles**2,)
    assert q.angle_y.shape == (angles**2,)
    assert q.angle_w.shape == (angles**2,)


@pytest.mark.parametrize("angles", LDFE_ANGLES)
def test_ldfe_xy_weights_sum_to_one(angles):
    q = ants.angular_xy(angles, method="ldfe")
    np.testing.assert_allclose(q.angle_w.sum(), 1.0, rtol=1e-12)


@pytest.mark.parametrize("angles", LDFE_ANGLES)
def test_ldfe_xy_on_unit_sphere(angles):
    # In 2D the upper-hemisphere xi is recovered from x and y.
    q = ants.angular_xy(angles, method="ldfe")
    assert np.all(q.angle_x**2 + q.angle_y**2 <= 1.0 + 1e-12)


@pytest.mark.parametrize("angles", LDFE_ANGLES)
def test_ldfe_xy_all_quadrants_present(angles):
    q = ants.angular_xy(angles, method="ldfe")
    assert np.any(q.angle_x > 0) and np.any(q.angle_x < 0)
    assert np.any(q.angle_y > 0) and np.any(q.angle_y < 0)


# ====================================================================
# angular_xy (method="ldfe") -- reflective ordering
# ====================================================================


@pytest.mark.parametrize("angles", [4, 8])
@pytest.mark.parametrize("bc_x", [[1, 0], [0, 1]])
def test_ldfe_xy_x_reflective_ordering(angles, bc_x):
    q = ants.angular_xy(angles, bc_x=bc_x, method="ldfe")
    _check_axis_ordering(q.angle_x, [q.angle_y], bc_x)


@pytest.mark.parametrize("angles", [4, 8])
@pytest.mark.parametrize("bc_y", [[1, 0], [0, 1]])
def test_ldfe_xy_y_reflective_ordering(angles, bc_y):
    q = ants.angular_xy(angles, bc_y=bc_y, method="ldfe")
    _check_axis_ordering(q.angle_y, [q.angle_x], bc_y)


# ====================================================================
# angular_xyz (method="ldfe") -- basic properties
# ====================================================================


@pytest.mark.parametrize("angles", LDFE_ANGLES)
def test_ldfe_xyz_shape(angles):
    q = ants.angular_xyz(angles, method="ldfe")
    assert q.angle_x.shape == (2 * angles**2,)
    assert q.angle_y.shape == (2 * angles**2,)
    assert q.angle_z.shape == (2 * angles**2,)
    assert q.angle_w.shape == (2 * angles**2,)


@pytest.mark.parametrize("angles", LDFE_ANGLES)
def test_ldfe_xyz_weights_sum_to_one(angles):
    q = ants.angular_xyz(angles, method="ldfe")
    np.testing.assert_allclose(q.angle_w.sum(), 1.0, rtol=1e-12)


@pytest.mark.parametrize("angles", LDFE_ANGLES)
def test_ldfe_xyz_on_unit_sphere(angles):
    q = ants.angular_xyz(angles, method="ldfe")
    norm = q.angle_x**2 + q.angle_y**2 + q.angle_z**2
    np.testing.assert_allclose(norm, 1.0, atol=1e-9)


@pytest.mark.parametrize("angles", LDFE_ANGLES)
def test_ldfe_xyz_all_signs_present(angles):
    q = ants.angular_xyz(angles, method="ldfe")
    assert np.any(q.angle_x > 0) and np.any(q.angle_x < 0)
    assert np.any(q.angle_y > 0) and np.any(q.angle_y < 0)
    assert np.any(q.angle_z > 0) and np.any(q.angle_z < 0)


# ====================================================================
# angular_xyz (method="ldfe") -- reflective ordering
# ====================================================================


@pytest.mark.parametrize("angles", [4, 8])
@pytest.mark.parametrize("bc_z", [[1, 0], [0, 1]])
def test_ldfe_xyz_z_reflective_ordering(angles, bc_z):
    q = ants.angular_xyz(angles, bc_z=bc_z, method="ldfe")
    _check_axis_ordering(q.angle_z, [q.angle_x, q.angle_y], bc_z)


# ====================================================================
# Regression against the published Appendix table (LDFE-SA-1)
# ====================================================================


def test_ldfe_sa1_matches_paper():
    # Jarrell and Adams M&C 2011, Table III (first-octant directions).
    expected = np.array(
        [
            [2.09769106510711e-01, 2.09769106510711e-01, 9.54983687770317e-01],
            [9.54983687770317e-01, 2.09769106510711e-01, 2.09769106510711e-01],
            [2.09769106510711e-01, 9.54983687770317e-01, 2.09769106510711e-01],
            [5.77350269189625e-01, 5.77350269189625e-01, 5.77350269189625e-01],
        ]
    )
    q = ants.angular_xyz(4, method="ldfe")
    directions = np.column_stack((q.angle_x, q.angle_y, q.angle_z))
    # Every published first-octant direction must appear in the full set.
    for row in expected:
        match = np.all(np.isclose(directions, row, atol=1e-12), axis=1)
        assert match.sum() == 1, f"missing first-octant direction {row}"


# ====================================================================
# Error handling
# ====================================================================


@pytest.mark.parametrize("angles", [2, 6, 10, 12])
def test_ldfe_invalid_angles_raises(angles):
    with pytest.raises(ValueError):
        ants.angular_xy(angles, method="ldfe")
    with pytest.raises(ValueError):
        ants.angular_xyz(angles, method="ldfe")


def test_product_remains_default():
    # Default method is unchanged; product set still works for non-LDFE counts.
    q = ants.angular_xy(10)
    assert q.angle_x.shape == (100,)
    np.testing.assert_allclose(q.angle_w.sum(), 1.0, rtol=1e-12)
