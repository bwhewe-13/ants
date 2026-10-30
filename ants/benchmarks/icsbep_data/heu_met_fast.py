########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# ICSBEP highly enriched uranium metal fast benchmarks (HEU-MET-FAST)
#
########################################################################

"""Benchmark models transcribed from the input decks in the ICSBEP
Handbook, with temperatures and k-effective from Sections 3.4 and 3.5 of
each evaluation. Number densities are in atoms/barn-cm.
"""

from ants.benchmarks.icsbep import Benchmark, register
from ants.benchmarks.icsbep_data.pu_met_fast import NATURAL_URANIUM

# Bare HEU sphere (Godiva)
HEU_MET_FAST_001 = register(
    Benchmark(
        name="HEU-MET-FAST-001",
        radii=[8.7407],
        regions=["fuel"],
        materials={"fuel": "92234 4.9184-4 92235 4.4994-2 92238 2.4984-3"},
        temperature=300.0,
        keff=1.000,
        keff_unc=0.001,
        reference="Rev. 2 (2002), Inputs/Spectra/KENO299/HMF001",
    )
)

# Godiva shell model (gaps between shells are void)
HEU_MET_FAST_001_SHELL = register(
    Benchmark(
        name="HEU-MET-FAST-001-SHELL",
        radii=[
            1.0216,
            1.0541,
            6.2809,
            6.2937,
            7.7527,
            7.7620,
            8.2527,
            8.2610,
            8.7062,
            8.7499,
        ],
        regions=[
            "shell-1",
            "void",
            "shell-2",
            "void",
            "shell-3",
            "void",
            "shell-4",
            "void",
            "shell-5",
            "shell-6",
        ],
        materials={
            "shell-1": "92234 4.9357-4 92235 4.4936-2 92238 2.7213-3",
            "shell-2": "92234 4.9357-4 92235 4.5244-2 92238 2.4168-3",
            "shell-3": "92234 4.9357-4 92235 4.5268-2 92238 2.3930-3",
            "shell-4": "92234 4.9357-4 92235 4.5090-2 92238 2.5690-3",
            "shell-5": "92234 4.9357-4 92235 4.5239-2 92238 2.4215-3",
            "shell-6": "92234 4.8974-4 92235 4.4874-2 92238 2.4169-3",
        },
        temperature=300.0,
        keff=1.000,
        keff_unc=0.001,
        reference="Rev. 2 (2002), Inputs/Spectra/KENO299/HMF001SM",
    )
)

# Flattop-25, natural uranium reflected
HEU_MET_FAST_028 = register(
    Benchmark(
        name="HEU-MET-FAST-028",
        radii=[6.1156, 24.1242],
        regions=["fuel", "reflector"],
        materials={
            "fuel": "92234 4.8869-4 92235 4.4482-2 92238 2.7038-3",
            "reflector": NATURAL_URANIUM,
        },
        temperature=293.0,
        keff=1.0000,
        keff_unc=0.0030,
        reference="Rev. 1 (1999), Inputs/Spectra/KENO299/HMF028",
    )
)
