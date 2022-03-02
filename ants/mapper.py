########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import os
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Mapper:

    def __init__(self, cells_x, cell_width, map_x, map_key):
        self.cells_x = cells_x
        self.cell_width = cell_width
        self.map_x = map_x
        self.map_key = map_key

    def save_map(self, filename):
        with open(filename, "wb") as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def load_map(filename):
        with open(filename, "rb") as inp:
            map_obj = pickle.load(inp)
        return map_obj

    def adjust_widths(self, new_cells_x):
        sections = [list(y) for x, y in itertools.groupby(self.map_x)]
        width_x = self.cells_x * self.cell_width
        new_cell_width = width_x / new_cells_x
        new_map_x = []
        boundaries = np.insert(np.cumsum([len(section) * self.cell_width \
            for section in sections]), 0, 0)
        splits = [slice(int(boundaries[ii]/new_cell_width), \
            int(boundaries[ii+1] / new_cell_width)) for ii in range(len(sections))]
        new_map_x = np.zeros((new_cells_x))
        for ii in range(len(sections)):
            new_map_x[splits[ii]] = sections[ii][0]
        # for section in sections:
        #     section_width = len(section) * self.cell_width
        #     new_map_x.append([section[0]] * int(section_width / new_cell_width))
        # new_map_x = [item for sublist in new_map_x for item in sublist]
        # Reassign
        self.map_x = np.array(new_map_x)
        self.cells_x = new_cells_x
        self.cell_width = new_cell_width


class MapperGUI(Gtk.Window):
    __allowed_symbols = [" ", ",", "-", "0", "1", "2", "3", "4", "5", \
                         "6", "7", "8", "9"]

    def __init__(self, cells_x, cell_width, materials):
        # Problem Inputs
        self.cells_x = cells_x
        self.cell_width = cell_width
        self.materials = materials
        # Generate GUI Window
        super().__init__(title="ANTS - Mapper")
        self.set_border_width(10)
        self.gui_grid = Gtk.Grid(column_spacing=25, row_spacing=10)
        self._generate_buttons()
        self._generate_titles()
        self._generate_materials()
        self.add(self.gui_grid)

    def _generate_buttons(self):
        compile_button = Gtk.Button(label="Compile")
        compile_button.connect("clicked", self._generate_mapping)
        self.gui_grid.attach(compile_button, 0, 0, 2, 1)
        save_button = Gtk.Button(label="Save")
        save_button.connect("clicked", self._save_object)
        self.gui_grid.attach_next_to(save_button, compile_button, \
                                     Gtk.PositionType.RIGHT, 2, 1)

    def _generate_titles(self):
        cells_label = Gtk.Label(xalign=0)
        cells_label.set_markup("<b>Spatial Cells: </b>{}".format( \
                                                       self.cells_x))
        self.gui_grid.attach(cells_label, 0, 1, 2, 1)
        cell_width_label = Gtk.Label(xalign=0)
        cell_width_label.set_markup("<b>Cell Width: </b>{}".format( \
                                                       self.cell_width))
        self.gui_grid.attach_next_to(cell_width_label, cells_label, \
                                     Gtk.PositionType.RIGHT, 2, 1)
        material_title = Gtk.Label(xalign=0.5)
        material_title.set_markup("<u>Material</u>")
        self.gui_grid.attach(material_title, 0, 2, 2, 1)
        cell_title = Gtk.Label(xalign=0.5)
        cell_title.set_markup("<u>Cell Number Range</u>")
        self.gui_grid.attach_next_to(cell_title, material_title, \
                            Gtk.PositionType.RIGHT, 2, 1)

    def _generate_materials(self):
        self.spatial_cells = []
        self.material_names = []
        for row, material in enumerate(self.materials):
            self.material_names.append(material)
            temp_name = material.replace("material-","").replace("-", " ")
            mat_label = Gtk.Label(label=temp_name, xalign=0)
            self.gui_grid.attach(mat_label, 0, row+5, 2, 1)
            cell_numbers = Gtk.Entry()
            self.spatial_cells.append(cell_numbers)
            self.gui_grid.attach_next_to(cell_numbers, mat_label, \
                                Gtk.PositionType.RIGHT, 2, 1)

    def _generate_mapping(self, widget):
        self.material_key = {}
        material_id = 0
        self.map_x = np.ones((self.cells_x)) * -1
        for name, cells in zip(self.material_names, self.spatial_cells):
            self.material_key[name] = material_id
            cell_locs = self._format_range(cells.get_text())
            assert np.all(self.map_x[cell_locs] == -1),\
                "Trying to overwrite allocated cell"
            self.map_x[cell_locs] = material_id
            material_id += 1
        self._generate_graph()
    
    def _save_object(self, action):
        save_dialog = Gtk.FileChooserDialog(title="Save Mapper", \
                     parent=None, action=Gtk.FileChooserAction.SAVE)
        save_dialog.add_buttons("_Save", Gtk.ResponseType.OK)
        save_dialog.add_buttons("_Cancel", Gtk.ResponseType.CLOSE)
        save_dialog.set_current_name("Untitled")
        save_dialog.set_do_overwrite_confirmation(True)
        save_dialog.set_local_only(True)
        response = save_dialog.run()
        if response == Gtk.ResponseType.OK:
            path = os.path.join(save_dialog.get_current_folder(), \
                                save_dialog.get_filename())
            map_obj = Mapper(self.cells_x, self.cell_width, self.map_x, \
                             self.material_key)
            map_obj.save_map(path + ".mpr")
        save_dialog.destroy()
        Gtk.main_quit()

    def _format_range(self, string):
        assert (set(string) <= set(self.__class__.__allowed_symbols)), \
            "{} Uses incorrect symbols. Limited to:\n{}".format(string,\
                                        self.__class__.__allowed_symbols)
        cell_index = []
        for split in string.split(','):
            if "-" in split:
                temp_size = split.split('-')
                assert len(temp_size) == 2
                cell_index.append(np.arange(int(temp_size[0])-1, \
                                            int(temp_size[1])))
            else:
                cell_index.append(np.array([int(split)-1]))
        return np.concatenate(cell_index)

    def _generate_graph(self):
        values = np.sort(np.unique(self.map_x))
        cmap = matplotlib.cm.jet.copy()
        if -1 in values:
            map_to_graph = np.ma.masked_where(self.map_x == -1, self.map_x)
            cmap.set_bad('gray')
        else:
            map_to_graph = self.map_x.copy()
        fig, ax = plt.subplots()
        graph = ax.imshow(np.tile(map_to_graph, (10, 1)), cmap=cmap)
        colors = [graph.cmap(graph.norm(value)) for value in values]
        patches = []
        if -1 in values:
            patches += [mpatches.Patch(color='gray', label='Void')]        
        patches += [mpatches.Patch(color=colors[vv], \
                      label=kk.replace("material-","").replace("-", " ")) \
                      for kk, vv in self.material_key.items() if vv != -1]
        plt.legend(handles=patches, framealpha=1, ncol=2, \
                 loc='upper center', bbox_to_anchor=(0.5, 2), fancybox=True)
        ax.set_yticks([])
        ax.set_xlabel('Distance (cm)')
        ax.set_xticks(np.linspace(0, self.cells_x, 5))
        ax.set_xticklabels(np.round(np.linspace(0, self.cells_x \
                            * self.cell_width, 5), 2).astype('str'))
        ax.grid(axis='x')
        plt.show()


def gui(cells_x, cell_width, materials):
    window = MapperGUI(cells_x, cell_width, materials)
    window.connect("destroy", Gtk.main_quit)
    window.show_all()
    Gtk.main()

if __name__ == "__main__":
    # materials = ["material-stainless-steel-440-%20%", \
    #              "material-high-density-polyethyene-618", \
    #              "high-density-polyethyene-087"]
    materials = ['material-reed-vacuum', 'material-reed-strong-source', \
                 'material-reed-scatter','material-reed-absorber']
    # materials = ['material-vacuum-reed']
    window = MapperGUI(160, 0.1, materials)
    window.connect("destroy", Gtk.main_quit)
    window.show_all()
    Gtk.main()

