########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Suite of ICSBEP fast metal sphere benchmarks (k-eigenvalue) using a
# HDF5 multigroup library, varying the angles and spatial cells.
#
# Usage: python icsbep_suite.py path/to/library.h5
#
########################################################################

import sys

import ants

library = ants.Library(sys.argv[1])
print(library)

# Bare and reflected fast spheres
names = ants.list_benchmarks("MET-FAST")

results = ants.run_suite(names, library, cells=[100, 400], angles=[4, 8, 16])

print(f"{'Benchmark':<24} {'G':>4} {'N':>4} {'I':>5} {'keff':>9} {'diff (pcm)':>11}")
for row in results:
    print(
        f"{row['name']:<24} {row['groups']:>4} {row['angles']:>4} "
        f"{row['cells']:>5} {row['keff']:>9.5f} {row['diff_pcm']:>11.0f}"
    )
