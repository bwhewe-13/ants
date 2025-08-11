# Running 2d multigroup problem - chevron

import numpy as np
import argparse
from memory_profiler import memory_usage

import ants
from ants import timed2d, critical2d
from ants.utils import hybrid as hytools

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--angles", type=int, action="store")
parser.add_argument("-g", "--groups", type=int, action="store")
args = parser.parse_args()

cells_x = 90
cells_y = 90
angles = args.angles
groups = args.groups

# Create labels
fgroups = str(groups).zfill(2)
fangles = str(angles).zfill(2)

steps = 50
T = 50e-6

steps = 1
T = 1e-6
dt = np.round(T / steps, 10)
edges_t = np.round(np.linspace(0, steps * dt, steps + 1), 10)


length_x = 9.
length_y = 9.

delta_x = np.repeat(length_x / cells_x, cells_x)
edges_x = np.round(np.linspace(0, length_x, cells_x+1), 10)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

delta_y = np.repeat(length_y / cells_y, cells_y)
edges_y = np.round(np.linspace(0, length_y, cells_y+1), 10)
centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

# Boundary conditions
bc_x = [0, 0]
bc_y = [0, 0]

# Velocity
edges_g, edges_gidx = ants.energy_grid(87, groups)
velocity = ants.energy_velocity(groups, edges_g)


# Cross Sections
materials = ["uranium-%0.7%", "high-density-polyethyene-087"]
xs_total, xs_scatter, xs_fission = ants.materials(87, materials)

# # Create chevrons
# triangle01 = [(0.1, 1.), (0.1, 3.9), (5.9, 1.)]
# triangle02 = [(6., 1.), (8.9, 1.), (8.9, 3.9)]
# triangle03 = [(0.1, 4.9), (0.1, 7.8), (5.9, 4.9)]
# triangle04 = [(6., 4.9), (8.9, 4.9), (8.9, 7.8)]
# triangles = np.array([triangle01, triangle02, triangle03, triangle04])
# t_index = [1, 1, 0, 0]

# # Create border
# rectangle01 = [(0, 0), 0.1, 9.]
# rectangle02 = [(0, 8.9), 9., 0.1]
# rectangle03 = [(8.9, 0), 0.1, 9.]
# rectangles = [rectangle01, rectangle02, rectangle03]
# r_index = [0, 0, 0]

# N_particles = cells_x * cells_y * 40
# weight_matrix = ants.weight_matrix2d(edges_x, edges_y, 3, N_particles=N_particles, \
#                                     triangles=triangles, triangle_index=t_index, \
#                                     rectangles=rectangles, rectangle_index=r_index)
# np.save("weight_matrix_g87_090", weight_matrix)

weight_matrix = np.load("weight_matrix_g87_090.npy")
data = ants.weight_spatial2d(weight_matrix, xs_total, xs_scatter, xs_fission)
medium_map, xs_total, xs_scatter, xs_fission = data

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(9, 9))
# img = ax.pcolormesh(weight_matrix[:,:,0].T, cmap="rainbow")
# fig.colorbar(img)
# ax.set_title("Chevron Problem")
# skip = 10
# ax.grid(True, axis="both", linestyle=":", color="k", alpha=0.6)
# ax.set_xticks(np.linspace(0, cells_x, cells_x+1)[::skip])
# ax.set_xticklabels(edges_x[::skip])
# ax.set_yticks(np.linspace(0, cells_y, cells_y+1)[::skip])
# ax.set_yticklabels(edges_y[::skip])
# ax.set_xlabel("x (cm)")
# ax.set_ylabel("y (cm)")
# ax.set_aspect("equal", "box")
# # fig.savefig("chevron/chevron_layout.png", bbox_inches="tight", dpi=200)
# plt.show()


info = {
            "cells_x": cells_x,
            "cells_y": cells_y,
            "angles": angles, 
            "groups": groups, 
            "materials": xs_total.shape[0],
            "geometry": 1, 
            "spatial": 2, 
            "bc_x": bc_x,
            "bc_y": bc_y, 
            "steps": steps,
            "dt": dt
        }


angle_x, angle_y, angle_w = ants.angular_xy(info)

# Boundary conditions and external source
external = np.zeros((1, cells_x, cells_y, 1, 1))

boundary_x, boundary_y = ants.boundary2d.deuterium_tritium(-1, 0, edges_g)
boundary_x = boundary_x[None,...].copy()

gamma_steps = ants.gamma_time_steps(edges_t)
boundary_y = ants.boundary2d.time_dependence_decay_03(boundary_y, gamma_steps)


if edges_g.shape[0] != (groups + 1):
    xs_total, xs_scatter, xs_fission = hytools.coarsen_materials(xs_total, \
                            xs_scatter, xs_fission, edges_g, edges_gidx)
    velocity = hytools.coarsen_velocity(velocity, edges_gidx)
    external = hytools.coarsen_external(external, edges_g, edges_gidx)
    boundary_x = hytools.coarsen_external(boundary_x, edges_g, edges_gidx)
    boundary_y = hytools.coarsen_external(boundary_y, edges_g, edges_gidx)



initial_x = np.zeros((cells_x + 1, cells_y, angles**2, groups))
initial_y = np.zeros((cells_x, cells_y + 1, angles**2, groups))


# data = {"xs_total": xs_total, "xs_scatter": xs_scatter, "xs_fission": xs_fission,
#         "velocity": velocity, "external": external, "boundary_x": boundary_x, 
#         "boundary_y": boundary_y, "medium_map": medium_map, "edges_x": edges_x,
#         "edges_y": edges_y, "angle_x": angle_x, "angle_y": angle_y,
#         "angle_w": angle_w, "info": info, "edges_g": edges_g[edges_gidx]}
# np.savez(f"data_g{fgroups}_n{fangles}.npz", **data)

def run():
    _ = timed2d.tr_bdf2(initial_x, initial_y, xs_total, xs_scatter, \
                        xs_fission, velocity, external, boundary_x, \
                        boundary_y, medium_map, delta_x, delta_y, \
                        angle_x, angle_y, angle_w, info)

# np.save(f"flux_mgsi_g{fgroups}_n{fangles}", flux)


# flux, keff = critical2d.power_iteration(xs_total, xs_scatter, xs_fission, \
#                                         medium_map, delta_x, delta_y, \
#                                         angle_x, angle_y, angle_w, info)

# mydic = {"flux": flux, "keff": keff}
# np.savez(f"flux_critical_g{gg}_n{nn}", **mydic)

if __name__ == "__main__":
    mem_usage = memory_usage((run, ()), interval=0.5)
    max_mem = np.max(mem_usage)
    diff_mem = np.max(mem_usage) - np.min(mem_usage)
    mean_mem = np.mean(mem_usage)
    std_mem = np.std(mem_usage)
    quantile = np.quantile(mem_usage, [0.0, 0.25, 0.5, 0.75, 1.0])
    with open("multigroup-memory-usage.txt", "a") as f:
        f.write(
            f"Angles: {fangles}, Groups: {fgroups}, "
            f"Max: {max_mem}, Diff: {diff_mem}, Mean: {mean_mem}, "
            f"Std: {std_mem}, Quantiles: {quantile}\n"
        )
    print(f"Max Memory: {max_mem:.2f} MB, Diff: {diff_mem:.2f} MB")