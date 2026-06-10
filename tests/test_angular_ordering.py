########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Tests for quadrature sweep ordering in angular_xy and angular_xyz.
#
# Sweep ordering convention: positive-direction angles sweep from the
# origin boundary outward; negative-direction angles sweep inward.
# When bc=[1,0] (reflect at origin), the negative direction must be
# processed before its paired positive direction so the reflected
# boundary flux is available. bc=[0,1] (reflect at far boundary)
# requires the opposite.
#
########################################################################

import numpy as np
import pytest

import ants

# ====================================================================
# Helper
# ====================================================================


def _check_axis_ordering(primary, others, bc):
    """Assert that every reflection pair along `primary` respects `bc`.

    A reflection pair is two directions that share the same values on
    all `others` axes and have opposite signs on `primary`.  For
    bc=[1,0] the negative-primary direction must appear at a lower
    array index than its positive-primary mirror; for bc=[0,1] the
    opposite.
    """
    n = len(primary)
    neg_first = bc == [1, 0]
    pair_count = 0

    for i in range(n):
        if abs(primary[i]) < 1e-10:
            continue
        for j in range(i + 1, n):
            if not np.isclose(primary[i], -primary[j], atol=1e-10):
                continue
            if not all(np.isclose(o[i], o[j], atol=1e-10) for o in others):
                continue
            # (i, j) is a reflection pair along the primary axis
            if neg_first:
                first = i if primary[i] < 0 else j
                second = j if primary[i] < 0 else i
            else:
                first = i if primary[i] > 0 else j
                second = j if primary[i] > 0 else i
            assert first < second, (
                f"Reflection pair ({i}, {j}): expected "
                f"{'negative' if neg_first else 'positive'}-primary "
                f"at index {first} < {second}"
            )
            pair_count += 1

    assert pair_count > 0, "No reflection pairs found — check quadrature generation"


# ====================================================================
# angular_xy — basic properties
# ====================================================================


@pytest.mark.parametrize("angles", [4, 8])
def test_angular_xy_shape(angles):
    q = ants.angular_xy(angles)
    assert q.angle_x.shape == (angles**2,)
    assert q.angle_y.shape == (angles**2,)
    assert q.angle_w.shape == (angles**2,)


@pytest.mark.parametrize("angles", [4, 8])
def test_angular_xy_weights_sum_to_one(angles):
    q = ants.angular_xy(angles)
    np.testing.assert_allclose(q.angle_w.sum(), 1.0, rtol=1e-12)


@pytest.mark.parametrize("angles", [4, 8])
def test_angular_xy_all_quadrants_present(angles):
    q = ants.angular_xy(angles)
    assert np.any(q.angle_x > 0) and np.any(q.angle_x < 0)
    assert np.any(q.angle_y > 0) and np.any(q.angle_y < 0)


# ====================================================================
# angular_xy — sweep ordering with reflective BCs
# ====================================================================


@pytest.mark.parametrize("angles", [4, 8])
@pytest.mark.parametrize("bc_x", [[1, 0], [0, 1]])
def test_angular_xy_x_reflective_ordering(angles, bc_x):
    """For each x-reflection pair (same angle_y, opposite angle_x),
    the required direction appears first."""
    q = ants.angular_xy(angles, bc_x=bc_x)
    _check_axis_ordering(q.angle_x, [q.angle_y], bc_x)


@pytest.mark.parametrize("angles", [4, 8])
@pytest.mark.parametrize("bc_y", [[1, 0], [0, 1]])
def test_angular_xy_y_reflective_ordering(angles, bc_y):
    """For each y-reflection pair (same angle_x, opposite angle_y),
    the required direction appears first."""
    q = ants.angular_xy(angles, bc_y=bc_y)
    _check_axis_ordering(q.angle_y, [q.angle_x], bc_y)


@pytest.mark.parametrize(
    "bc_x,bc_y",
    [
        ([1, 0], [1, 0]),
        ([1, 0], [0, 1]),
        ([0, 1], [1, 0]),
        ([0, 1], [0, 1]),
    ],
)
def test_angular_xy_both_axes_reflective_ordering(bc_x, bc_y):
    """Both x and y constraints are satisfied simultaneously."""
    q = ants.angular_xy(4, bc_x=bc_x, bc_y=bc_y)
    _check_axis_ordering(q.angle_x, [q.angle_y], bc_x)
    _check_axis_ordering(q.angle_y, [q.angle_x], bc_y)


# ====================================================================
# angular_xyz — basic properties
# ====================================================================


@pytest.mark.parametrize("angles", [4, 8])
def test_angular_xyz_shape(angles):
    q = ants.angular_xyz(angles)
    assert q.angle_x.shape == (2 * angles**2,)
    assert q.angle_y.shape == (2 * angles**2,)
    assert q.angle_z.shape == (2 * angles**2,)
    assert q.angle_w.shape == (2 * angles**2,)


@pytest.mark.parametrize("angles", [4, 8])
def test_angular_xyz_weights_sum_to_one(angles):
    q = ants.angular_xyz(angles)
    np.testing.assert_allclose(q.angle_w.sum(), 1.0, rtol=1e-12)


@pytest.mark.parametrize("angles", [4, 8])
def test_angular_xyz_all_signs_present(angles):
    q = ants.angular_xyz(angles)
    assert np.any(q.angle_x > 0) and np.any(q.angle_x < 0)
    assert np.any(q.angle_y > 0) and np.any(q.angle_y < 0)
    assert np.any(q.angle_z > 0) and np.any(q.angle_z < 0)


# ====================================================================
# angular_xyz — sweep ordering with reflective BCs
# ====================================================================


@pytest.mark.parametrize("angles", [4, 8])
@pytest.mark.parametrize("bc_x", [[1, 0], [0, 1]])
def test_angular_xyz_x_reflective_ordering(angles, bc_x):
    """For each x-reflection pair (same angle_y and angle_z, opposite angle_x),
    the required direction appears first."""
    q = ants.angular_xyz(angles, bc_x=bc_x)
    _check_axis_ordering(q.angle_x, [q.angle_y, q.angle_z], bc_x)


@pytest.mark.parametrize("angles", [4, 8])
@pytest.mark.parametrize("bc_y", [[1, 0], [0, 1]])
def test_angular_xyz_y_reflective_ordering(angles, bc_y):
    """For each y-reflection pair (same angle_x and angle_z, opposite angle_y),
    the required direction appears first."""
    q = ants.angular_xyz(angles, bc_y=bc_y)
    _check_axis_ordering(q.angle_y, [q.angle_x, q.angle_z], bc_y)


@pytest.mark.parametrize("angles", [4, 8])
@pytest.mark.parametrize("bc_z", [[1, 0], [0, 1]])
def test_angular_xyz_z_reflective_ordering(angles, bc_z):
    """For each z-reflection pair (same angle_x and angle_y, opposite angle_z),
    the required direction appears first."""
    q = ants.angular_xyz(angles, bc_z=bc_z)
    _check_axis_ordering(q.angle_z, [q.angle_x, q.angle_y], bc_z)


@pytest.mark.parametrize(
    "bc_x,bc_y,bc_z",
    [
        ([1, 0], [1, 0], [1, 0]),
        ([1, 0], [0, 1], [0, 0]),
        ([0, 1], [1, 0], [1, 0]),
        ([0, 0], [0, 1], [1, 0]),
        ([0, 1], [0, 1], [0, 1]),
    ],
)
def test_angular_xyz_multi_axis_reflective_ordering(bc_x, bc_y, bc_z):
    """All active reflective constraints are satisfied simultaneously."""
    q = ants.angular_xyz(4, bc_x=bc_x, bc_y=bc_y, bc_z=bc_z)
    if bc_x != [0, 0]:
        _check_axis_ordering(q.angle_x, [q.angle_y, q.angle_z], bc_x)
    if bc_y != [0, 0]:
        _check_axis_ordering(q.angle_y, [q.angle_x, q.angle_z], bc_y)
    if bc_z != [0, 0]:
        _check_axis_ordering(q.angle_z, [q.angle_x, q.angle_y], bc_z)
