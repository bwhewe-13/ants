########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Kobayashi dog leg benchmark (Problem 3) with 50% scattering. A source
# in the corner of a shield streams through a void duct with three bends.
# The x = 0, y = 0, and z = 0 faces are reflecting. The flux is compared
# with the Monte Carlo reference along the three lines of the benchmark.
# The same problem is an example in the MC/DC repository
# (examples/kobayashi-dogleg).
#
########################################################################

import matplotlib.pyplot as plt
import numpy as np

from ants.benchmarks import kobayashi
from ants.datatypes import SolverData
from ants.fixed3d import fixed_source

problem = 3
scattering = True
cell_width = 2.0
angles = 8

# Artificial scattering reduces the ray effects far from the source, but
# the lagged source takes much longer to converge in the duct
sigma_as = 0.0

solver = SolverData(num_threads=0, sigma_as=sigma_as)
inputs = kobayashi.build_problem(problem, scattering, cell_width, angles, solver=solver)
geometry = inputs[2]

flux = fixed_source(*inputs)
flux = flux[..., 0]

# Each line of reference points changes along one axis
lines = {
    "3A": (1, "y (cm)", "x = z = 5 cm"),
    "3B": (0, "x (cm)", "y = 55 cm, z = 5 cm"),
    "3C": (0, "x (cm)", "y = 95 cm, z = 35 cm"),
}

fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="col")
for col, (case, (axis, label, title)) in enumerate(lines.items()):
    points, reference, uncertainty = kobayashi.reference(case, scattering)
    approx = kobayashi.flux_at(flux, points, geometry)
    distance = points[:, axis]
    error = np.max(np.fabs(approx / reference - 1.0))
    print(f"{case}: largest difference from the reference {error:.1%}")

    axes[0, col].errorbar(
        distance,
        reference,
        yerr=2 * uncertainty * reference,
        fmt="o",
        c="k",
        label="Reference (GMVP)",
    )
    axes[0, col].semilogy(distance, approx, "-s", c="r", label=f"ANTS S{angles}")
    axes[0, col].set_title(f"{case}: {title}")
    axes[0, col].grid(which="both", alpha=0.3)

    axes[1, col].plot(distance, approx / reference, "-s", c="r")
    axes[1, col].axhline(1.0, c="k", ls="--")
    axes[1, col].set_xlabel(label)
    axes[1, col].grid(alpha=0.3)

axes[0, 0].set_ylabel("Total flux (cm$^{-2}$ s$^{-1}$)")
axes[1, 0].set_ylabel("ANTS / Reference")
axes[0, 0].legend(loc=0, framealpha=1)
fig.tight_layout()
plt.show()

# np.save("kobayashi_problem_3", flux)
