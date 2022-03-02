########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
#
########################################################################

import ants

class Transport(Materials):

    def __init__(self, materials, energy_groups, energy_bounds,\
                 energy_idx=None, filename=None, map_key=None):
        self.filename = filename
        self.map_key = map_key
        super(MapMaterials, self).__init__(materials, energy_groups, \
                                        energy_bounds, energy_idx=None)
        self.compile_map_materials()

    def compile_map_materials(self):
        map_obj = mapper.Mapper.load_map(self.filename)
        self.map_x = map_obj.map_x
        if self.map_key is None:
            self.map_key = map_obj.map_key
        reversed_key = {v: k for k, v in self.map_key.items()}
        self.total = []
        self.scatter = []
        self.fission = []
        for position in range(len(self.map_key)):
            map_material = reversed_key[position]
            self.total.append(self.data[map_material][0])
            self.scatter.append(self.data[map_material][1])
            self.fission.append(self.data[map_material][2])