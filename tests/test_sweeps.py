########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
# 
# Test Problems for constucting material cross sections for materials 
# listed below. The cross sections are compared to the original one-
# dimenionsal 'discrete1' package.
# 
# Materials: "uranium", "uranium-hydride", "plutonium" 
#            "stainless-steel-440", "high-density-polyethyene-618", 
#            "high-density-polyethyene-087", "carbon", "uranium-235", 
#            "uranium-238", "water-uranium-dioxide", "plutonium-239", 
#            "plutonium-240"
#
########################################################################

from ants.criticality import keigenvalue

import pytest
import numpy as np

@pytest.mark.smoke
@pytest.mark.one_group
@pytest.mark.slab
def test_plutonium_239_01():
    groups = 1
    angles = 8
    cells = 100

    rad = 1.853722 * 2
    cell_width = rad / cells


    xs_fission = np.array([[[3.24 * 0.0816]]])
    xs_total = np.array([[0.32640]])
    xs_scatter = np.array([[[0.225216]]])
    medium_map = np.zeros((cells),dtype=int)

