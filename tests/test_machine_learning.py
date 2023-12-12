########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Test Machine Learning Functions
#
########################################################################

import pytest
import numpy as np
from sklearn import metrics

from ants.utils import machine_learning as ml


np.random.seed(42)


@pytest.mark.math
def test_mean_absolute_error():
    # Create 2 different arrays
    n = 100
    y_true = np.sort(2 * np.random.rand(n, n))
    y_pred = y_true + 0.1 * np.random.rand(n, n)
    # Get sklearn function
    reference = []
    for row in range(n):
        reference.append(metrics.mean_absolute_error(y_true[row], y_pred[row]))
    reference = np.array(reference)
    # Personal implementation
    approx = ml.mean_absolute_error(y_true, y_pred)
    assert np.array_equal(reference, approx), "Implementation doesn't match SKLearn"


@pytest.mark.math
def test_mean_squared_error():
    # Create 2 different arrays
    n = 100
    y_true = np.sort(2 * np.random.rand(n, n))
    y_pred = y_true + 0.1 * np.random.rand(n, n)
    # Get sklearn function
    reference = []
    for row in range(n):
        reference.append(metrics.mean_squared_error(y_true[row], y_pred[row]))
    reference = np.array(reference)
    # Personal implementation
    approx = ml.mean_squared_error(y_true, y_pred)
    assert np.array_equal(reference, approx), "Implementation doesn't match SKLearn"


@pytest.mark.math
def test_explained_variance_score():
    # Create 2 different arrays
    n = 100
    y_true = np.sort(2 * np.random.rand(n, n))
    y_pred = y_true + 0.1 * np.random.rand(n, n)
    # Get sklearn function
    reference = []
    for row in range(n):
        reference.append(metrics.explained_variance_score(y_true[row], y_pred[row]))
    reference = np.array(reference)
    # Personal implementation
    approx = ml.explained_variance_score(y_true, y_pred)
    assert np.array_equal(reference, approx), "Implementation doesn't match SKLearn"


@pytest.mark.math
def test_r2_score():
    # Create 2 different arrays
    n = 100
    y_true = np.sort(2 * np.random.rand(n, n))
    y_pred = y_true + 0.1 * np.random.rand(n, n)
    # Get sklearn function
    reference = []
    for row in range(n):
        reference.append(metrics.r2_score(y_true[row], y_pred[row]))
    reference = np.array(reference)
    # Personal implementation
    approx = ml.r2_score(y_true, y_pred)
    assert np.array_equal(reference, approx), "Implementation doesn't match SKLearn"