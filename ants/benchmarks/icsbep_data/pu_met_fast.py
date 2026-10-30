########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# ICSBEP plutonium metal fast benchmarks (PU-MET-FAST)
#
########################################################################

"""Benchmark models transcribed from the simplified input decks in the
ICSBEP Handbook, with temperatures and k-effective from Sections 3.4 and
3.5 of each evaluation. Number densities are in atoms/barn-cm; ZAIDs
ending in 000 are natural elements.
"""

from ants.benchmarks.icsbep import Benchmark, register

# Natural uranium reflector (Flattop)
NATURAL_URANIUM = """
    92234 2.6438-6
    92235 3.4610-4
    92238 4.7721-2
"""

# Bare Pu-239 Jezebel
PU_MET_FAST_001 = register(
    Benchmark(
        name="PU-MET-FAST-001",
        radii=[6.39157],
        regions=["fuel"],
        materials={
            "fuel": """
                "94239.700nm"  0.037047
                "94240.700nm"  0.0017512
                "94241.700nm"  0.00011674
                "31069.700nm"  0.00083603
                "31071.700nm"  0.00053917
            """,
        },
        temperature=293.15,
        keff=1.00000,
        keff_unc=0.00129,
        reference="Rev. 3 (2013), Inputs/PARTISN/Simplified/simple_model_30g",
    )
)

# Bare Pu-240 Jezebel
PU_MET_FAST_002 = register(
    Benchmark(
        name="PU-MET-FAST-002",
        radii=[6.6595],
        regions=["fuel"],
        materials={
            "fuel": """
                94239.55c 0.029934
                94240.50c 0.0078754
                94241.50c 0.0012146
                94242.50c 0.00015672
                31000.50c 0.0013722
            """,
        },
        temperature=293.15,
        keff=1.0000,
        keff_unc=0.0020,
        reference="Rev. 1 (1999), Inputs/MCNP/CASE_1",
    )
)

# Flattop-Pu, natural uranium reflected
PU_MET_FAST_006 = register(
    Benchmark(
        name="PU-MET-FAST-006",
        radii=[4.5332, 24.142],
        regions=["fuel", "reflector"],
        materials={
            "fuel": """
                94239 3.6697-2
                94240 1.8700-3
                94241 1.1639-4
                31000 1.4755-3
            """,
            "reflector": NATURAL_URANIUM,
        },
        temperature=293.0,
        keff=1.0000,
        keff_unc=0.0030,
        reference="Rev. 1 (1999), Inputs/Spectra/KENO299/PMF06001",
    )
)

# Delta phase plutonium, thorium reflected
PU_MET_FAST_008 = register(
    Benchmark(
        name="PU-MET-FAST-008",
        radii=[5.310, 29.88],
        regions=["fuel", "reflector"],
        materials={
            "fuel": """
                94239.55c 0.036049
                94240.50c 0.0019562
                94241.50c 0.00011459
                31000.50c 0.0013338
            """,
            "reflector": "90232.50c 0.030054",
        },
        temperature=293.0,
        keff=1.0000,
        keff_unc=0.0006,
        reference="Rev. 2 (1999), Inputs/MCNP/CASE_1",
    )
)

# Delta phase plutonium, beryllium oxide reflected
PU_MET_FAST_018 = register(
    Benchmark(
        name="PU-MET-FAST-018",
        radii=[5.0419, 8.7300],
        regions=["fuel", "reflector"],
        materials={
            "fuel": """
                94239.55c 0.037291
                94240.50c 0.0019277
                94241.50c 0.00012196
                31000.50c 0.0013628
            """,
            "reflector": """
                4009.50c 0.11984
                8016.50c 0.0013776
            """,
        },
        temperature=293.0,
        keff=1.0000,
        keff_unc=0.0030,
        reference="Inputs/MCNP/CASE_1",
    )
)
