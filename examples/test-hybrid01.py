import discrete1
import numpy as np

import ants
from ants.utils import hybrid as hytools

# General conditions
cells_x = 1000
angles_u = 8
angles_c = 2
groups_u = 87
groups_c = 43
steps = 5

print("Groups", groups_u, groups_c)
print("Angles", angles_u, angles_c)

info_u = {
    "cells_x": cells_x,
    "angles": angles_u,
    "groups": groups_u,
    "materials": 2,
    "geometry": 1,
    "spatial": 2,
    "bc_x": [0, 0],
    "steps": steps,
    "dt": 1e-8,
}

info_c = {
    "cells_x": cells_x,
    "angles": angles_c,
    "groups": groups_c,
    "materials": 2,
    "geometry": 1,
    "spatial": 2,
    "bc_x": [0, 0],
    "steps": steps,
    "dt": 1e-8,
}


# Spatial
length = 10.0
delta_x = np.repeat(length / cells_x, cells_x)
edges_x = np.linspace(0, length, cells_x + 1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Energy Grid
edges_g, edges_gidx_u, edges_gidx_c = ants.energy_grid(87, groups_u, groups_c)
velocity_u = ants.energy_velocity(groups_u, edges_g)
velocity_c = hytools.coarsen_velocity(velocity_u, edges_gidx_c)

# Angular
angle_xu, angle_wu = ants.angular_x(info_u)
angle_xc, angle_wc = ants.angular_x(info_c)

# Medium Map
layers = [[0, "stainless-steel-440", "0-4, 6-10"], [1, "uranium-%20%", "4-6"]]
medium_map = ants.spatial1d(layers, edges_x)

# Cross Sections - Uncollided
materials = np.array(layers)[:, 1]
xs_total_u, xs_scatter_u, xs_fission_u = ants.materials(87, materials)
# Cross Sections - Collided
xs_collided = hytools.coarsen_materials(
    xs_total_u, xs_scatter_u, xs_fission_u, edges_g[edges_gidx_u], edges_gidx_c
)
xs_total_c, xs_scatter_c, xs_fission_c = xs_collided

# External and boundary sources
external = np.zeros((1, cells_x, 1, 1))
boundary_x = ants.boundary1d.deuterium_tritium(0, edges_g)
edges_t = np.linspace(0, steps * info_u["dt"], steps + 1)
boundary_x = ants.boundary1d.time_dependence_decay_02(boundary_x, edges_t)


# Indexing Parameters
fine_idx, coarse_idx, factor = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
# import discrete1
# edges_g, edges_gidx_u, edges_gidx_c = discrete1.energy_grid(87, groups_u, groups_c)
# print(coarse_idx)

# print(coarse_idx.shape, edges_gidx_c.shape)
# print(edges_gidx_c)

# print(edges_gidx_c.shape, edges_gidx_u.shape)
# print(edges_gidx_c)
# print(edges_gidx_u)

# print(edges_g.shape)
# print("edges_gidx_u", edges_gidx_u, edges_gidx_u.shape)
# print("fine_idx", fine_idx, fine_idx.shape)
# print("edges_gidx_c", edges_gidx_c, edges_gidx_c.shape)
# print("coarse idx", coarse_idx, coarse_idx.shape)
# print(factor.shape)

initial_flux = np.zeros((cells_x, angles_u, groups_u))

# # Run Hybrid Method
# flux = backward_euler(initial_flux, xs_total_u, xs_total_c, xs_scatter_u, \
#             xs_scatter_c, xs_fission_u, xs_fission_c, velocity_u, \
#             velocity_c, external, boundary_x, medium_map, delta_x, \
#             angle_xu, angle_xc, angle_wu, angle_wc, fine_idx, coarse_idx, \
#             factor, info_u, info_c)

# np.save(
#     f"hybrid_uranium_slab_g{groups_u}g{groups_c}_n{angles_u}n{angles_c}_flux",
#     flux,
# )

# np.random.seed(3)
# flux = np.random.rand(cells_x, groups_u)


# def first(flux_u, xs_scatter, source_c, coarse_idx):

#     # Zero out previous source
#     source_c *= 0.0

#     # Iterate over all spatial cells
#     # for ii in range(flux.shape[0]):
#     ii = 0
#     mat = medium_map[ii]
#     for og in range(flux_u.shape[1]):
#         for ig in range(flux_u.shape[1]):
#             source_c[ii, 0, coarse_idx[og]] += (
#                 flux_u[ii, ig] * xs_scatter[mat, og, ig]
#             )
#             # source_c[ii, 0, coarse_idx[og]] += 1
#         # print(coarse_idx[og])

#     # return source_c


# def second(flux_u, xs_scatter, source_c, edges_gidx_c):
#     # Zero out previous source
#     source_c *= 0.0

#     # Iterate over all spatial cells
#     # for ii in range(flux_u.shape[0]):
#     ii = 0
#     mat = medium_map[ii]
#     for gg in range(edges_gidx_c.shape[0] - 1):
#         source = 0.0
#         for og in range(edges_gidx_c[gg], edges_gidx_c[gg + 1]):
#             #     for ig in range(edges_gidx_c[gg], edges_gidx_c[gg + 1]):
#             for ig in range(flux_u.shape[1]):
#                 # for ig in range(flux_u.shape[1]):
#                 # print(og, ig)
#                 source += flux_u[ii, ig] * xs_scatter[mat, og, ig]
#                 # source += 1
#         source_c[ii, 0, gg] += source

# return source_c


# source1 = np.zeros((cells_x, 1, groups_c))
# source2 = np.zeros((cells_x, 1, groups_c))

# second(flux, xs_scatter_u, source2, edges_gidx_c)
# first(flux, xs_scatter_u, source1, coarse_idx)


# print(np.sum(source1))
# print(np.sum(source2))
# print(np.array_equal(source1, source2), np.sum(np.fabs(source1 - source2)))


# def first(flux_u, flux_c, edges_gidx_c):
#     groups_c = flux_c.shape[1]
#     flux_c *= 0.0
#     # Iterate over collided groups
#     for gg in range(groups_c):
#         flux_c[:, gg] = np.sum(
#             flux_u[:, edges_gidx_c[gg] : edges_gidx_c[gg + 1]], axis=1
#         )


# def second(flux_u, flux_c, edges_gidx_c):
#     cells_x, groups_c = flux_c.shape
#     tmp_flux = 0.0
#     flux_c *= 0.0

#     for ii in range(cells_x):
#         for og in range(groups_c):
#             tmp_flux = 0.0
#             for ig in range(edges_gidx_c[og], edges_gidx_c[og + 1]):
#                 tmp_flux += flux_u[ii, ig]
#             flux_c[ii, og] = tmp_flux


# import discrete1
# edges_g, edges_gidx_u, edges_gidx_c = discrete1.energy_grid(87, groups_u, groups_c)

# np.random.seed(3)
# flux = np.random.rand(cells_x, groups_u)


# print(flux.shape)
# print(edges_gidx_c.shape)

# flux1 = np.zeros((cells_x, groups_c))
# flux2 = np.zeros((cells_x, groups_c))

# first(flux, flux1, edges_gidx_c)
# print(np.sum(flux1), np.sum(flux2))


# second(flux, flux2, edges_gidx_c)
# print(np.array_equal(flux1, flux2), np.sum(flux1), np.sum(flux2))
# def first(xs_total_u, delta_fine, delta_coarse, star_coef_c, edges_gidx_c, gg):
#     idx1 = edges_gidx_c[gg]
#     idx2 = edges_gidx_c[gg + 1]

#     xs_total_c = (
#         np.sum(xs_total_u[:, idx1:idx2] * delta_fine[idx1:idx2], axis=1)
#         / delta_coarse[gg]
#     )
#     xs_total_c += star_coef_c

#     # xs_total_c = (
#     #     np.sum(
#     #         xs_total_u[:, idx1:idx2, idx1:idx2] * delta_fine[idx1:idx2],
#     #         axis=(1, 2),
#     #     )
#     #     / delta_coarse[gg]
#     # )

#     return xs_total_c


# def second(xs_total_u, star_coef_c, edges_g, delta_coarse, edges_gidx_c, gg):
#     idx1 = edges_gidx_c[gg]
#     idx2 = edges_gidx_c[gg + 1]
#     xs_total_c = np.zeros((2,))
#     delta_coarse = 1 / delta_coarse

#     for mat in range(xs_total_u.shape[0]):
#         for og in range(idx1, idx2):
#             xs_total_c[mat] += xs_total_u[mat, og] * (edges_g[og + 1] - edges_g[og])
#         # xs_total_c[mat] = xs_total_c[mat] / delta_coarse + star_coef_c
#         xs_total_c[mat] = delta_coarse
#         xs_total_c[mat] += star_coef_c

#         # for ii in range(idx1, idx2):
#         #     for jj in range(idx1, idx2):
#         #         xs_total_c[mat] += xs_total_u[mat, ii, jj] * (
#         #             edges_g[jj + 1] - edges_g[jj]
#         #         )
#         # xs_total_c[mat] *= delta_coarse

#     return xs_total_c


# xs_total_c1 = np.zeros((len(materials), groups_c))
# xs_total_c2 = np.zeros((len(materials), groups_c))

# delta_fine = np.diff(edges_g[edges_gidx_u])
# delta_coarse = np.diff(edges_g[edges_gidx_u][edges_gidx_c])

# star_coef_c = 0.5
# gg = 41

# xs_total_c1 = first(
#     xs_total_u, delta_fine, delta_coarse, star_coef_c, edges_gidx_c, gg
# )

# xs_total_c2 = second(
#     xs_total_u,
#     star_coef_c,
#     edges_g,
#     edges_g[edges_gidx_c[gg + 1]] - edges_g[edges_gidx_c[gg]],
#     edges_gidx_c,
#     gg,
# )

# # print(xs_total_u)
# print(xs_total_c1)
# print(xs_total_c2)


# def first(flux_orig, xs_scatter_u, delta_coarse, delta_fine, groups_c, gg):
#     flux = flux_orig / delta_coarse
#     # flux = flux_orig.copy()

#     xs_scatter = xs_scatter_u * delta_fine

#     off_scatter = np.zeros((cells_x,))
#     for og in range(groups_c):

#         idx1 = edges_gidx_c[og]
#         idx2 = edges_gidx_c[og + 1]

#         if og > gg:
#             for ii in range(cells_x):
#                 mat = medium_map[ii]
#                 off_scatter[ii] += (
#                     np.sum(xs_scatter[mat, :, idx1:idx2]) * flux[ii, og]
#                 )
#     return off_scatter


# def second(
#     flux_orig, xs_scatter, edges_g, edges_gidx_c, groups_c, out_idx1, out_idx2, gg
# ):

#     flux = flux_orig.copy()

#     off_scatter = np.zeros((cells_x,))
#     for og in range(groups_c):

#         idx1 = edges_gidx_c[og]
#         idx2 = edges_gidx_c[og + 1]

#         if og > gg:
#             for ii in range(cells_x):
#                 mat = medium_map[ii]
#                 flux_tmp = 0.0
#                 for aa in range(out_idx1, out_idx2):
#                     for bb in range(idx1, idx2):
#                         flux_tmp += (
#                             xs_scatter[mat, aa, bb]
#                             * (edges_g[bb + 1] - edges_g[bb])
#                             * flux[ii, og]
#                             / (edges_g[idx2] - edges_g[idx1])
#                         )
#                 off_scatter[ii] = flux_tmp

#     return off_scatter


# gg = 41
# out_idx1 = edges_gidx_c[gg]
# out_idx2 = edges_gidx_c[gg + 1]

# np.random.seed(3)
# flux_c = 0.5 * np.ones((cells_x, groups_c)) + np.random.rand(cells_x, groups_c)

# delta_fine = np.diff(edges_g[edges_gidx_u])
# delta_coarse = np.diff(edges_g[edges_gidx_u][edges_gidx_c])

# off_scatter1 = first(
#     flux_c, xs_scatter_u[:, out_idx1:out_idx2], delta_coarse, delta_fine, groups_c, gg
# )

# off_scatter2 = second(
#     flux_c, xs_scatter_u, edges_g, edges_gidx_c, groups_c, out_idx1, out_idx2, gg
# )

# print(
#     np.array_equal(off_scatter1, off_scatter2),
#     np.sum(np.fabs(off_scatter1 - off_scatter2)),
#     np.sum(off_scatter1),
#     np.sum(off_scatter2),
# )


def first(flux_u, flux_c, xs_matrix_u, medium_map, coarse_idx, factor_u):

    # Initialize iterables
    one_group = 0.0

    source = np.zeros((cells_x, angles_u, groups_u))
    flux = np.zeros(flux_u.shape)

    # Assume that source is already (Qu + 1 / (v * dt) * psi^{\ell-1})
    for ii in range(flux_u.shape[0]):
        mat = medium_map[ii]

        for og in range(flux_u.shape[1]):
            flux[ii, og] = flux_u[ii, og] + flux_c[ii, coarse_idx[og]] * factor_u[og]

            # if ii == 1:
            #     print(og, factor_u[og])

        for og in range(flux_u.shape[1]):
            one_group = 0.0
            for ig in range(flux_u.shape[1]):
                one_group += flux[ii, ig] * xs_matrix_u[mat, og, ig]
            for nn in range(source.shape[1]):
                source[ii, nn, og] += one_group

    return source, flux


def second(flux_u, flux_c, xs_matrix_u, medium_map, edges_g, edges_gidx_c):

    # Initialize iterables
    one_group = 0.0

    source = np.zeros((cells_x, angles_u, groups_u))
    flux = np.zeros(flux_u.shape)

    # Assume that source is already (Qu + 1 / (v * dt) * psi^{\ell-1})
    for ii in range(flux_u.shape[0]):
        mat = medium_map[ii]

        for og in range(flux_c.shape[1]):
            idx1 = edges_gidx_c[og]
            idx2 = edges_gidx_c[og + 1]
            delta_coarse = 1.0 / (edges_g[idx2] - edges_g[idx1])

            for ig in range(idx1, idx2):
                flux[ii, ig] = (
                    flux_u[ii, ig]
                    + flux_c[ii, og] * (edges_g[ig + 1] - edges_g[ig]) * delta_coarse
                )

                # if ii == 0:
                #     print(ig, (edges_g[ig + 1] - edges_g[ig]) * delta_coarse)

        #     flux_u[ii, og] = (
        #         flux_u[ii, og] + flux_c[ii, coarse_idx[og]] * factor_u[og]
        #     )

        for og in range(flux_u.shape[1]):
            one_group = 0.0
            for ig in range(flux_u.shape[1]):
                one_group += flux[ii, ig] * xs_matrix_u[mat, og, ig]
            for nn in range(source.shape[1]):
                source[ii, nn, og] += one_group

    return source, flux


fine_idx, coarse_idx, factor = hytools.indexing(edges_g, edges_gidx_u, edges_gidx_c)
edges_g, edges_gidx_u, edges_gidx_c = discrete1.energy_grid(87, groups_u, groups_c)
delta_fine = np.diff(edges_g[edges_gidx_u])
delta_coarse = np.diff(edges_g[edges_gidx_u][edges_gidx_c])

np.random.seed(3)
flux_u = 0.5 * np.ones((cells_x, groups_u)) + np.random.rand(cells_x, groups_u)
flux_c = 0.75 * np.ones((cells_x, groups_c)) + np.random.rand(cells_x, groups_c)
source = 0.25 * np.ones((cells_x, angles_u, groups_u)) + np.random.rand(
    cells_x, angles_u, groups_u
)
# print(coarse_idx)

# print(edges_gidx_c)
# print(factor)
xs_matrix = xs_scatter_u + xs_fission_u

source1, flux1 = first(flux_u, flux_c, xs_matrix, medium_map, coarse_idx, factor)
source2, flux2 = second(flux_u, flux_c, xs_matrix, medium_map, edges_g, edges_gidx_c)

# print(np.sum(flux1))
# print(np.sum(flux2))
print(np.array_equal(flux1, flux2), np.sum(np.fabs(flux1 - flux2)))

print(np.sum(source1))
print(np.sum(source2))
print(np.array_equal(source1, source2), np.sum(np.fabs(source1 - source2)))
