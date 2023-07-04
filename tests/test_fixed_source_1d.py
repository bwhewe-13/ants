########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Test one-dimensional time-independent problems
#
########################################################################

import pytest
import numpy as np

import ants
from ants import fixed1d
from tests import problems1d


@pytest.mark.sphere1d
@pytest.mark.source_iteration
@pytest.mark.multigroup1d
def test_sphere_01_source_iteration():
    PATH = problems1d.sphere_01("fixed")[-1]
    flux = fixed1d.source_iteration(*problems1d.sphere_01("fixed")[:-1])
    reference = np.load(PATH + "uranium_sphere_source_iteration_flux.npy")
    assert np.isclose(flux, reference).all()