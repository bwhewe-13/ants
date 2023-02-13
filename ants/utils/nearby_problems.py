########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
# Running the Method of Nearby Problems for both fixed source and
# criticality problems.
#
########################################################################

from ants.utils import dimensions, interp1d
from ants.utils.interp1d import QuinticHermite as Hermite
from ants import fixed1d, critical1d

import numpy as np

def refine_interface_grid(medium_map, delta_x, dtype="half"):
    mat_splits = dimensions.create_slices(medium_map)
    updated_delta_x = []
    for count, split in enumerate(mat_splits):
        temp = delta_x[split].copy()
        if dtype == "half":
            if count != (len(mat_splits) - 1):
                temp = np.append(temp[:-1], 2 * [temp[-1] * 0.5])
            if count != 0:
                temp = np.append(2 * [temp[0] * 0.5], temp[1:])
        elif dtype == "step":
            if count != (len(mat_splits) - 1):
                step_down = np.concatenate((2 * [temp[-2] * 0.75], 3 * [temp[-1] * 0.5]))
                temp = np.append(temp[:-3], np.round(step_down, 10))
            if count != 0:
                step_down = np.concatenate((3 * [temp[0] * 0.5], 2 * [temp[0] * 0.75]))
                temp = np.append(np.round(step_down, 10), temp[3:])
        updated_delta_x.append(temp)
    updated_delta_x = np.array([ii for sublst in updated_delta_x for ii in sublst])
    return updated_delta_x

def reaction_rates(flux, xs, medium_map):
    reaction = np.zeros((len(flux)))
    for cell in range(len(medium_map)):
        mat = medium_map[cell]
        reaction[cell] = np.sum(flux[cell] * np.sum(xs[mat], axis=1))
    return reaction

def update_medium_map(medium_map, edges, new_edges):
    if len(new_edges) > len(edges):
        return _expand_map(medium_map, edges, new_edges)
    elif len(new_edges) < len(edges):
        return _shrink_map(medium_map, edges, new_edges)
    return _adjust_map(medium_map, edges, new_edges)

def _expand_map(medium_map, edges, new_edges):
    new_centers = 0.5 * (new_edges[1:] + new_edges[:-1])
    new_medium_map = -1 * np.ones((len(new_centers)), dtype=np.int32)
    for cell, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        idx = np.argwhere((new_centers < right) & (new_centers > left))
        new_medium_map[idx.flatten()] = medium_map[cell]
    return new_medium_map

def _shrink_map(medium_map, edges, new_edges):
    centers = 0.5 * (edges[1:] + edges[:-1])
    new_medium_map = -1 * np.ones((len(new_edges)-1), dtype=np.int32)
    for cell, (left, right) in enumerate(zip(new_edges[:-1], new_edges[1:])):
        idx = np.argwhere((centers < right) & (centers > left))
        new_medium_map[cell] = np.unique(medium_map[idx.flatten()])
    return new_medium_map

def _adjust_map(medium_map, edges, new_edges):
    new_medium_map = -1 * medium_map.copy()
    raise Exception("Not Finished!")

class Critical:

    def __init__(self, xs_total, xs_scatter, xs_fission, medium_map, \
                 delta_x, angle_x, angle_w, params):
        self.xs_total = xs_total
        self.xs_scatter = xs_scatter
        self.xs_fission = xs_fission
        self.medium_map = medium_map
        self.delta_x = delta_x
        self.angle_x = angle_x
        self.angle_w = angle_w
        self.params = params.copy()

    def run(self, dtype="point"):
        self.edges_x = np.insert(np.cumsum(self.delta_x), 0, 0.)
        self.centers_x = 0.5 * (self.edges_x[1:] + self.edges_x[:-1])
        self._critical()
        self._numerical_fission_source()
        if dtype == "point":
            self._curve_fit_point()
            self._residual_point()
        else:
            self._curve_fit()
            self._residual(dtype)
        self._nearby(dtype)
        self._normalize()

    def _critical(self):
        self.critical_flux, self.numerical_keff = critical1d.power_iteration( \
            self.xs_total, self.xs_scatter, self.xs_fission, self.medium_map, \
            self.delta_x, self.angle_x, self.angle_w, self.params)

    def _numerical_fission_source(self):
        self.numerical_source = np.zeros((self.params["cells"], self.params["groups"]))
        for cell, mat in enumerate(self.medium_map):
            self.numerical_source[cell] += self.xs_fission[mat] @ self.critical_flux[cell]
        self.numerical_source /= self.numerical_keff
        boundary = np.zeros((2,))
        self.numerical_flux = fixed1d.source_iteration(self.xs_total, self.xs_scatter, \
            self.xs_fission * 0.0, self.numerical_source.flatten(), boundary, \
            self.medium_map, self.delta_x, self.angle_x, self.angle_w, self.params)

    def _curve_fit_point(self):
        splits = dimensions.create_slices(self.medium_map)
        self.curve_fit_flux = np.zeros(self.numerical_flux.shape)
        self.curve_fit_dflux = np.zeros(self.numerical_flux.shape)
        self.curve_fit_iflux = np.zeros(self.numerical_flux.shape)
        for group in range(self.params["groups"]):
            for angle in range(self.params["angles"]):
                for split in splits:
                    func = Hermite(self.centers_x[split], self.numerical_flux[split,angle,group])
                    self.curve_fit_flux[split,angle,group] = func(self.centers_x[split])
                    self.curve_fit_dflux[split,angle,group] = func.derivative()(self.centers_x[split])
                    self.curve_fit_iflux[split,angle,group] = func.integrate(\
                                        self.edges_x[split.start:split.stop+1])

    def _curve_fit_keffective(self):
        LHS = np.zeros(self.curve_fit_flux.shape)
        RHS = np.zeros(self.curve_fit_flux.shape)
        scalar_flux = np.sum(self.curve_fit_iflux * self.angle_w[None,:,None], axis=1)
        boundary = np.zeros((2, self.params["angles"], self.params["groups"]))
        dstream = np.diff(dimensions.flux_edges(self.curve_fit_flux, self.angle_x, boundary), axis=0)
        for group in range(self.params["groups"]):
            for angle in range(self.params["angles"]):
                for cell, mat in enumerate(self.medium_map):
                    LHS[cell,angle,group] = (self.angle_x[angle] * dstream[cell,angle,group] \
                        + self.curve_fit_iflux[cell,angle,group] * self.xs_total[mat][group]) \
                        - (scalar_flux[cell] @ self.xs_scatter[mat].T)[group]
                    RHS[cell,angle,group] = (scalar_flux[cell] @ self.xs_fission[mat].T)[group]
        LHS = np.sum(LHS * self.angle_w[None,:,None], axis=1)
        RHS = np.sum(RHS * self.angle_w[None,:,None], axis=1)
        return np.sum(RHS) / np.sum(LHS)

    def _residual_point(self):
        self.curve_fit_keff = self._curve_fit_keffective()
        self.curve_fit_source = self._fission_soure_point()
        self.curve_fit_source /= self.curve_fit_keff
        self.residual = np.zeros(self.numerical_flux.shape)
        scalar_flux = np.sum(self.curve_fit_flux * self.angle_w[None,:,None], axis=1)
        for group in range(self.params["groups"]):
            for angle in range(self.params["angles"]):
                for cell, mat in enumerate(self.medium_map):
                    self.residual[cell,angle,group] = (self.angle_x[angle] * self.curve_fit_dflux[cell,angle,group] \
                        + self.curve_fit_flux[cell,angle,group] * self.xs_total[mat][group]) \
                        - (scalar_flux[cell] @ self.xs_scatter[mat].T)[group] \
                        - self.curve_fit_source[group::self.params["groups"]][cell]

    def _fission_soure_point(self):
        fission_rate = 0.0
        curve_fit_source = np.zeros((self.params["cells"], self.params["groups"]))
        scalar_flux = np.sum(self.curve_fit_flux * self.angle_w[None,:,None], axis=1)
        for cell, mat in enumerate(self.medium_map):
            fission_rate += np.sum(self.xs_fission[mat] @ scalar_flux[cell]) / self.delta_x[cell]
            curve_fit_source[cell] = self.xs_fission[mat] @ scalar_flux[cell]
        self.nearby_rate = self.curve_fit_keff / fission_rate
        return curve_fit_source.flatten()

    def _curve_fit(self):
        splits = dimensions.create_slices(self.medium_map)
        self.curve_fit_flux = np.zeros(self.numerical_flux.shape)
        for group in range(self.params["groups"]):
            for angle in range(self.params["angles"]):
                for split in splits:
                    func = Hermite(self.centers_x[split], self.numerical_flux[split,angle,group])
                    self.curve_fit_flux[split,angle,group] = func(self.centers_x[split])

    def _curve_fit_fission_source(self, dtype="centers"):
        rhs = np.zeros(self.curve_fit_flux.shape)
        lhs = np.zeros(self.curve_fit_flux.shape)
        splines, dsplines = self._integrate_splines(dtype)
        ssplines = np.sum(splines * self.angle_w[None,:,None], axis=1)
        for gg in range(self.params["groups"]):
            for nn in range(self.params["angles"]):
                for ii, mat in enumerate(self.medium_map):
                    rhs[ii,nn,gg] = (ssplines[ii] @ self.xs_fission[mat].T)[gg]
                    lhs[ii,nn,gg] = (self.angle_x[nn] * dsplines[ii,nn,gg] \
                            + splines[ii,nn,gg] * self.xs_total[mat,gg]) \
                            - (ssplines[ii] @ self.xs_scatter[mat].T)[gg]
        rhs = np.sum(rhs * self.angle_w[None,:,None], axis=1)
        lhs = np.sum(lhs * self.angle_w[None,:,None], axis=1)
        self.curve_fit_keff = np.sum(rhs) / np.sum(lhs)
        # print("Curve fit keff", self.curve_fit_keff)
        curve_fit_source = np.zeros((self.params["cells"], self.params["groups"]))
        self.nearby_rate = 0.0
        for cell, mat in enumerate(self.medium_map):
            curve_fit_source[cell] += self.xs_fission[mat] @ ssplines[cell]
            self.nearby_rate += np.sum(self.xs_fission[mat] @ ssplines[cell]) / self.delta_x[cell]
        return curve_fit_source / self.curve_fit_keff

    def _residual(self, dtype="centers"):
        self.residual = np.zeros(self.numerical_flux.shape) # (I x N x G)
        self.curve_fit_source = self._curve_fit_fission_source(dtype)
        splines, dsplines = self._integrate_splines(dtype)
        # Calculate scalar flux
        ssplines = np.sum(splines * self.angle_w[None,:,None], axis=1)
        for gg in range(self.params["groups"]):
            for nn in range(self.params["angles"]):
                for ii, mat in enumerate(self.medium_map):
                    self.residual[ii,nn,gg] = (self.angle_x[nn] * dsplines[ii,nn,gg] \
                            + splines[ii,nn,gg] * self.xs_total[mat,gg]) \
                            - (ssplines[ii] @ self.xs_scatter[mat].T)[gg] \
                            - self.curve_fit_source[ii,gg] #* self.delta_x[ii]

    def _integrate_splines(self, dtype):
        edges_splits = dimensions.create_slices(self.medium_map, double_count=True)
        centers_splits = dimensions.create_slices(self.medium_map, double_count=False)
        splines = np.zeros((self.params["cells"], self.params["angles"], self.params["groups"]))
        dsplines = np.zeros(splines.shape)
        if dtype == "centers":
            for csplit, esplit in zip(centers_splits, edges_splits):
                splines[csplit], dsplines[csplit] = Hermite.integrate_splines_centers( \
                        self.edges_x[esplit], self.numerical_flux[csplit], self.params)
        elif dtype == "edges":
            boundary = np.zeros((2, self.params["angles"], self.params["groups"]))
            flux = dimensions.flux_edges(self.numerical_flux, self.angle_x, boundary)
            for csplit, esplit in zip(centers_splits, edges_splits):
                splines[csplit], dsplines[csplit] = Hermite.integrate_splines_edges( \
                        self.edges_x[esplit], flux[esplit], self.params)
        return splines, dsplines

    def _nearby(self, dtype):
        print("dtype:", dtype)
        self.params["qdim"] = 3
        # mydic = {"xs_total": self.xs_total, "xs_scatter": self.xs_scatter,
        #         "xs_fission": self.xs_fission, "residual": self.residual.flatten(), \
        #         "medium_map": self.medium_map, "delta_x": self.delta_x, \
        #         "angle_x": self.angle_x, "angle_w": self.angle_w, \
        #         "nearby_rate": self.nearby_rate, "params" :self.params}
        self.nearby_flux, self.nearby_keff = critical1d.nearby_power( \
                self.xs_total, self.xs_scatter, self.xs_fission, \
                self.residual.flatten(), self.medium_map, self.delta_x, \
                self.angle_x, self.angle_w, self.nearby_rate, self.params)
        # print("Nearby Keff:", self.nearby_keff)

    def _normalize(self):
        self.nearby_flux /= np.linalg.norm(self.nearby_flux)
        self.curve_fit_flux = np.sum(self.curve_fit_flux * self.angle_w[None,:,None], axis=1)
        self.curve_fit_flux /= np.linalg.norm(self.curve_fit_flux)

class FixedSource:

    def __init__(self, xs_total, xs_scatter, xs_fission, source, boundary, \
                 medium_map, delta_x, angle_x, angle_w, params):
        self.xs_total = xs_total
        self.xs_scatter = xs_scatter
        self.xs_fission = xs_fission
        self.source = source
        self.boundary = boundary
        self.medium_map = medium_map
        self.delta_x = delta_x
        self.angle_x = angle_x
        self.angle_w = angle_w
        self.params = params

    def run(self, dtype="point"):
        self.edges_x = np.insert(np.cumsum(self.delta_x), 0, 0.)
        self.centers_x = 0.5 * (self.edges_x[1:] + self.edges_x[:-1])
        self._numerical()
        # pointinal Method
        if dtype == "point":
            self._curve_fit_point()
            self._residual_point()
        # New Method
        else:
            self._curve_fit()
            self._residual(dtype)
        self._nearby(dtype)

    def _numerical(self):
        self.numerical_flux = fixed1d.source_iteration(self.xs_total, self.xs_scatter, \
            self.xs_fission, self.source.flatten(), self.boundary.flatten(), \
            self.medium_map, self.delta_x, self.angle_x, self.angle_w, self.params)

    def _curve_fit_point(self):
        splits = dimensions.create_slices(self.medium_map)
        self.curve_fit_flux = np.zeros(self.numerical_flux.shape)
        self.curve_fit_dflux = np.zeros(self.numerical_flux.shape)
        self.curve_fit_boundary = np.zeros((2,) + self.numerical_flux.shape[1:])
        for group in range(self.params["groups"]):
            for angle in range(self.params["angles"]):
                for cc, split in enumerate(splits):
                    func = Hermite(self.centers_x[split], self.numerical_flux[split,angle,group])
                    self.curve_fit_flux[:,angle,group][split] = func(self.centers_x[split])
                    self.curve_fit_dflux[:,angle,group][split] = func.derivative()(self.centers_x[split])
                    if cc == 0:
                        self.curve_fit_boundary[0,angle,group] = func([self.edges_x[0]])
                    if cc == (len(splits) - 1):
                        self.curve_fit_boundary[1,angle,group] = func([self.edges_x[-1]])

    def _residual_point(self):
        self.residual = np.zeros(self.source.shape)
        scalar_flux = np.sum(self.curve_fit_flux * self.angle_w[None,:,None], axis=1)
        for group in range(self.params["groups"]):
            for angle in range(self.params["angles"]):
                for cell, mat in enumerate(self.medium_map):
                    self.residual[cell,angle,group] = (self.angle_x[angle] \
                            * self.curve_fit_dflux[cell,angle,group]
                            + self.curve_fit_flux[cell,angle,group] * self.xs_total[mat,group]) \
                            - (scalar_flux[cell] @ self.xs_scatter[mat].T)[group] \
                            - (scalar_flux[cell] @ self.xs_fission[mat].T)[group] \
                            - self.source[cell,angle,group]
        # print("Residual Sum:", np.sum(self.residual))

    def _curve_fit(self):
        splits = dimensions.create_slices(self.medium_map)
        self.curve_fit_flux = np.zeros(self.numerical_flux.shape)
        self.curve_fit_boundary = np.zeros((2,) + self.numerical_flux.shape[1:])
        for group in range(self.params["groups"]):
            for angle in range(self.params["angles"]):
                for cc, split in enumerate(splits):
                    func = Hermite(self.centers_x[split], self.numerical_flux[:,angle,group][split])
                    self.curve_fit_flux[:,angle,group][split] = func(self.centers_x[split])
                    if cc == 0:
                        self.curve_fit_boundary[0,angle,group] = func([self.edges_x[0]])
                    if cc == (len(splits) - 1):
                        self.curve_fit_boundary[1,angle,group] = func([self.edges_x[-1]])

    def _residual(self, dtype="centers"):
        self.residual = np.zeros(self.source.shape)
        splines, dsplines = self._integrate_splines(dtype)
        # Calculate scalar flux
        ssplines = np.sum(splines * self.angle_w[None,:,None], axis=1)
        for gg in range(self.params["groups"]):
            for nn in range(self.params["angles"]):
                for ii, mat in enumerate(self.medium_map):
                    self.residual[ii,nn,gg] = (self.angle_x[nn] * dsplines[ii,nn,gg] \
                            + splines[ii,nn,gg] * self.xs_total[mat,gg]) \
                            - (ssplines[ii] @ self.xs_scatter[mat].T)[gg] \
                            - (ssplines[ii] @ self.xs_fission[mat].T)[gg] \
                            - self.source[ii,nn,gg] * self.delta_x[ii]

    def _integrate_splines(self, dtype):
        edges_splits = dimensions.create_slices(self.medium_map, double_count=True)
        centers_splits = dimensions.create_slices(self.medium_map, double_count=False)
        splines = np.zeros(self.source.shape)
        dsplines = np.zeros(splines.shape)
        if dtype == "centers":
            for csplit, esplit in zip(centers_splits, edges_splits):
                splines[csplit], dsplines[csplit] = Hermite.integrate_splines_centers( \
                        self.edges_x[esplit], self.numerical_flux[csplit], self.params)
        elif dtype == "edges":
            flux = dimensions.flux_edges(self.numerical_flux, self.angle_x, self.boundary)
            for csplit, esplit in zip(centers_splits, edges_splits):
                splines[csplit], dsplines[csplit] = Hermite.integrate_splines_edges( \
                        self.edges_x[esplit], flux[esplit], self.params)
        return splines, dsplines

    def _nearby(self, dtype):
        # Add residual to source term
        # if dtype == "point":
        #     print("saved")
        #     np.save("temp-residual", self.residual)
        # if dtype in ["edges", "centers"]:
        #     self.residual *= 3
        #     print("loaded")
        #     self.residual = np.load("temp-residual.npy")
        updated_source = (self.source + self.residual).flatten()
        print("{} Residual Sum: {}".format(dtype, np.sum(self.residual)))
        self.nearby_flux = fixed1d.source_iteration(self.xs_total, self.xs_scatter, \
            self.xs_fission, updated_source, self.curve_fit_boundary.flatten(), \
            self.medium_map, self.delta_x, self.angle_x, self.angle_w, self.params)
        # Calculate Relative Error
        self.error = np.fabs(self.nearby_flux - self.curve_fit_flux) \
                        / self.curve_fit_flux
        self.error[np.isnan(self.error)] = 0.0

