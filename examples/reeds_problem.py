from ants.medium import MediumX
from ants.materials import Materials
from ants.mapper import Mapper
from ants.multi_group import source_iteration


import numpy as np
import matplotlib.pyplot as plt

def reeds(cells):
    width = 16.
    delta_x = width/cells
    group = 1

    boundaries = [slice(0,int(2/delta_x)),slice(int(2/delta_x),int(3/delta_x)),
        slice(int(3/delta_x),int(5/delta_x)),slice(int(5/delta_x),int(6/delta_x)),
        slice(int(6/delta_x),int(10/delta_x)),slice(int(10/delta_x),int(11/delta_x)),
        slice(int(11/delta_x),int(13/delta_x)),slice(int(13/delta_x),int(14/delta_x)),
        slice(int(14/delta_x),int(16/delta_x))]

    total_xs = np.zeros((cells,group),dtype='float64')
    total_vals = [10,10,0,5,50,5,0,10,10]
    # total_vals = [1,1,0,5,50,5,0,1,1]

    scatter_xs = np.zeros((cells,group,group),dtype='float64')
    scatter_vals = [9.9,9.9,0,0,0,0,0,9.9,9.9]
    # scatter_vals = [0.9,0.9,0,0,0,0,0,0.9,0.9]

    source = np.zeros((cells,group),dtype='float64')
    source_vals = [0,1,0,0,50,0,0,1,0]

    for ii in range(len(boundaries)):
        total_xs[boundaries[ii]] = total_vals[ii]
        scatter_xs[boundaries[ii]] = np.diag(np.repeat(scatter_vals[ii],group))
        source[boundaries[ii]] = source_vals[ii]
    
    # scatter_xs = np.ones((cells,group,group),dtype='float64') * 0.1
    return total_xs, scatter_xs, source

groups = 1
cells_x = 1000
medium_width = 16.
cell_width_x = medium_width / cells_x
angles = 16
xbounds = np.array([1, 0])

materials = ['reed-vacuum', 'reed-strong-source', \
                 'reed-scatter','reed-absorber']

problem_01 = Materials(materials, 1, None)
medium = MediumX(cells_x, cell_width_x, angles, xbounds)
medium.add_external_source("reed")

map_obj = Mapper.load_map('reed_problem2.mpr')
if cells_x != map_obj.cells_x:
    map_obj.adjust_widths(cells_x)



reversed_key = {v: k for k, v in map_obj.map_key.items()}
total = []
scatter = []
fission = []

for position in range(len(map_obj.map_key)):
    map_material = reversed_key[position]
    total.append(problem_01.data[map_material][0])
    scatter.append(problem_01.data[map_material][1])
    fission.append(problem_01.data[map_material][2])

total = np.array(total)
scatter = np.array(scatter)
fission = np.array(fission)

print(map_obj.map_key.keys())
print(problem_01.data.keys())

mu_x = medium.mu_x
weight = medium.weight

print(mu_x)
print(weight)

medium_map = map_obj.map_x.astype(int)

phi = source_iteration(groups, mu_x / cell_width_x, weight, total, scatter, \
                     fission, medium.ex_source, medium_map, xbounds, \
                     cell_width_x)

print(medium.ex_source.shape)

fig, ax = plt.subplots()
solution = np.load('reed_solution.npy')
print(len(solution))
print(np.allclose(solution, phi[:,0],atol=1e-12))

ax.plot(np.linspace(0, 16, len(solution)), solution, label='solution', c='k', ls='--')
ax.plot(np.linspace(0, medium_width, cells_x), phi[:,0], label='New', c='r', alpha=0.6)
ax.legend(loc=0)

plt.show()