########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# ICSBEP benchmark data (registered on import)
#
########################################################################

"""ICSBEP benchmark specifications.

The benchmark data are taken from the International Handbook of
Evaluated Criticality Safety Benchmark Experiments, NEA/NSC/DOC(95)03,
OECD Nuclear Energy Agency, September 2015 edition. The ``reference``
field of each benchmark gives the evaluation section and model used.
The included experiments (Jezebel, Godiva, Flattop, Thor, ...) were
performed at Los Alamos National Laboratory, and the evaluations are
credited to their authors in the Handbook.

Only the geometry, number densities, temperatures, and benchmark
k-effective values are included here. The full evaluations, including
uncertainties, experimental descriptions, and input decks, are
available from the OECD Nuclear Energy Agency. Publications that use
these benchmarks should cite the Handbook.
"""

from ants.benchmarks.icsbep_data import heu_met_fast  # noqa: F401
from ants.benchmarks.icsbep_data import pu_met_fast  # noqa: F401
from ants.benchmarks.icsbep_data import u233_met_fast  # noqa: F401
