# Running 2d hybrid problem - chevron

import numpy as np
import argparse
import psutil
import os
from memory_profiler import memory_usage

import ants
from ants import hybrid2d
from ants.utils import hybrid as hytools


parser = argparse.ArgumentParser()
parser.add_argument("-nu", "--angles_u", type=int, action="store", default=24)
parser.add_argument("-nc", "--angles_c", type=int, action="store")
parser.add_argument("-gu", "--groups_u", type=int, action="store", default=87)
parser.add_argument("-gc", "--groups_c", type=int, action="store")
args = parser.parse_args()

# General conditions
cells_x = 90
cells_y = 90

angles_u = args.angles_u
angles_c = args.angles_c

groups_u = args.groups_u
groups_c = args.groups_c

fangles_u = str(angles_u).zfill(2) 
fangles_c = str(angles_c).zfill(2)

fgroups_u = str(groups_u).zfill(2) 
fgroups_c = str(groups_c).zfill(2)


steps = 50
T = 50e-6

dt = np.round(T / steps, 10)
edges_t = np.round(np.linspace(0, steps * dt, steps + 1), 10)


# Spatial Layout
length_x = 9.
length_y = 9.

delta_x = np.repeat(length_x / cells_x, cells_x)
edges_x = np.linspace(0, length_x, cells_x+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

delta_y = np.repeat(length_y / cells_y, cells_y)
edges_y = np.linspace(0, length_y, cells_y+1)
centers_y = 0.5 * (edges_y[1:] + edges_y[:-1])

# Boundary conditions
bc_x = [0, 0]
bc_y = [0, 0]

# Energy Grid
edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(87, groups_u, groups_c)


# Cross Sections - Uncollided
materials = ["uranium-%0.7%", "high-density-polyethyene-087"]
xs_total_u, xs_scatter_u, xs_fission_u = ants.materials(87, materials)
velocity_u = ants.energy_velocity(groups_u, edges_g)

# Boundary conditions and external source
external = np.zeros((1, cells_x, cells_y, 1, 1))
boundary_x, boundary_y = ants.boundary2d.deuterium_tritium(-1, 0, edges_g)
boundary_x = boundary_x[None,...].copy()

gamma_steps = ants.gamma_time_steps(edges_t)
# gamma_steps = edges_t.copy()
boundary_y = ants.boundary2d.time_dependence_decay_03(boundary_y, gamma_steps)


if edges_g.shape[0] != (groups_u + 1):
    xs_total_u, xs_scatter_u, xs_fission_u = hytools.coarsen_materials(xs_total_u, \
                            xs_scatter_u, xs_fission_u, edges_g, edges_gidx_u)
    velocity_u = hytools.coarsen_velocity(velocity_u, edges_gidx_u)
    external = hytools.coarsen_external(external, edges_g, edges_gidx_u)
    boundary_x = hytools.coarsen_external(boundary_x, edges_g, edges_gidx_u)
    boundary_y = hytools.coarsen_external(boundary_y, edges_g, edges_gidx_u)


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
# Update cross sections for cylinder
data = ants.weight_spatial2d(weight_matrix, xs_total_u, xs_scatter_u, xs_fission_u)
medium_map, xs_total_u, xs_scatter_u, xs_fission_u = data

# Cross Sections - Collided
xs_collided = hytools.coarsen_materials(xs_total_u, xs_scatter_u, xs_fission_u, \
                                        edges_g[edges_gidx_u], edges_gidx_c)
xs_total_c, xs_scatter_c, xs_fission_c = xs_collided
velocity_c = hytools.coarsen_velocity(velocity_u, edges_gidx_c)

info_u = {
            "cells_x": cells_x,
            "cells_y": cells_y,
            "angles": angles_u,
            "groups": groups_u,
            "materials": xs_total_u.shape[0],
            "geometry": 1,
            "spatial": 2,
            "bc_x": bc_x,
            "bc_y": bc_y,
            "steps": steps,
            "dt": dt
        }

info_c = {
            "cells_x": cells_x,
            "cells_y": cells_y,
            "angles": angles_c,
            "groups": groups_c,
            "materials": xs_total_c.shape[0],
            "geometry": 1,
            "spatial": 2,
            "bc_x": bc_x,
            "bc_y": bc_y,
            "steps": steps,
            "dt": dt
        }


angle_xu, angle_yu, angle_wu = ants.angular_xy(info_u)
angle_xc, angle_yc, angle_wc = ants.angular_xy(info_c)


# Hybrid indexing
fine_idx, coarse_idx, factor = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)

# Initialize flux
initial_x = np.zeros((cells_x + 1, cells_y, angles_u**2, groups_u))
initial_y = np.zeros((cells_x, cells_y + 1, angles_u**2, groups_u))

# # inner psutil function
# def process_memory():
#     process = psutil.Process(os.getpid())
#     mem_info = process.memory_info()
#     return mem_info


# # decorator function
# def profile(func):
#     def wrapper(*args, **kwargs):

#         mem_info = process_memory()
#         rss_mem_before = mem_info.rss
#         vms_mem_before = mem_info.vms
        
#         result = func(*args, **kwargs)

#         mem_info = process_memory()
#         rss_mem_after = mem_info.rss
#         vms_mem_after = mem_info.vms

#         vms_mem_diff = (vms_mem_after - vms_mem_before) / (1024 * 1024)
#         rss_mem_diff = (rss_mem_after - rss_mem_before) / (1024 * 1024)

#         with open("hybrid-memory-usage.txt", "a") as f:
#             f.write(
#                 f"Angles: {fangles_u} {fangles_c}, Groups: {fgroups_u} {fgroups_c}, "
#                 f"Virtual Memory (MB): {vms_mem_diff}, RSS (MB): {rss_mem_diff}\n"
#             )
#         print(f"Virtual Memory: {vms_mem_diff:.2f} MB, RSS: {rss_mem_diff:.2f}")
#         return result

#     return wrapper

# Run Hybrid
# @profile
def run():
    _ = hybrid2d.tr_bdf2(initial_x, initial_y, xs_total_u, xs_total_c, \
                    xs_scatter_u, xs_scatter_c, xs_fission_u, xs_fission_c, \
                    velocity_u, velocity_c, external, boundary_x, boundary_y, \
                    medium_map, delta_x, delta_y, angle_xu, angle_xc, angle_yu, \
                    angle_yc, angle_wu, angle_wc, fine_idx, coarse_idx, factor, \
                    info_u, info_c)



# flux = hybrid2d.tr_bdf2(initial_x, initial_y, xs_total_u, xs_total_c, \
#                 xs_scatter_u, xs_scatter_c, xs_fission_u, xs_fission_c, \
#                 velocity_u, velocity_c, external, boundary_x, boundary_y, \
#                 medium_map, delta_x, delta_y, angle_xu, angle_xc, angle_yu, \
#                 angle_yc, angle_wu, angle_wc, fine_idx, coarse_idx, factor, \
#                 info_u, info_c)

# Create labels
# parameters = f"g{fgroups_u}g{fgroups_c}_n{fangles_u}n{fangles_c}"
# np.save(f"flux_hysi_" + parameters, flux)

if __name__ == "__main__":
    mem_usage = memory_usage((run, ()), interval=0.5)
    max_mem = np.max(mem_usage)
    diff_mem = np.max(mem_usage) - np.min(mem_usage)
    mean_mem = np.mean(mem_usage)
    std_mem = np.std(mem_usage)
    quantile = np.quantile(mem_usage, [0.0, 0.25, 0.5, 0.75, 1.0])
    with open("hybrid-memory-usage.txt", "a") as f:
        f.write(
            f"Angles: {fangles_u} {fangles_c}, Groups: {fgroups_u} {fgroups_c}, "
            f"Max: {max_mem}, Diff: {diff_mem}, Mean: {mean_mem}, "
            f"Std: {std_mem}, Quantiles: {quantile}\n"
        )
    print(f"Max Memory: {max_mem:.2f} MB, Diff: {diff_mem:.2f} MB")
