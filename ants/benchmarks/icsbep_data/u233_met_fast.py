########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# ICSBEP uranium-233 metal fast benchmarks (U233-MET-FAST)
#
########################################################################

"""Benchmark models transcribed from the input decks in the ICSBEP
Handbook, with temperatures and k-effective from Sections 3.4 and 3.5 of
each evaluation. Number densities are in atoms/barn-cm.
"""

from ants.benchmarks.icsbep import Benchmark, register

# Bare U-233 Jezebel
U233_MET_FAST_001 = register(
    Benchmark(
        name="U233-MET-FAST-001",
        radii=[5.9838],
        regions=["fuel"],
        materials={
            "fuel": """
                92233.50c 0.046712
                92234.50c 0.00059026
                92235.50c 0.000014281
                92238.50c 0.00028561
            """,
        },
        temperature=293.0,
        keff=1.000,
        keff_unc=0.001,
        reference="Sections 3.2-3.3, Inputs/Spectra/KENO299/UMF001",
    )
)

# Flattop-23, natural uranium reflected
U233_MET_FAST_006 = register(
    Benchmark(
        name="U233-MET-FAST-006",
        radii=[4.2058, 24.1194],
        regions=["fuel", "reflector"],
        materials={
            "fuel": """
                92233.50c 0.046710
                92234.50c 0.00058772
                92235.50c 0.000014158
                92238.50c 0.00027959
            """,
            "reflector": """
                92235.50c 0.00035050
                92238.50c 0.047719
            """,
        },
        temperature=293.0,
        keff=1.0000,
        keff_unc=0.0014,
        reference="Rev. 1 (2000), Sections 3.2-3.3, Inputs/Spectra/KENO299/UMF006",
    )
)
