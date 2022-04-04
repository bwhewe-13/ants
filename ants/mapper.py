########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

import numpy as np
import os
import itertools
import platform
import argparse
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import tkinter as tk
from tkinter import ttk
import tkinter.font as font
import tkinter.messagebox as messagebox
import tkinter.filedialog as filedialog
import tkinter.simpledialog as simpledialog

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


class MapperGUI(tk.Tk):
    __allowed_symbols = [" ", ",", "-", "0", "1", "2", "3", "4", "5", \
                 "6", "7", "8", "9"]

    _font = ("Arial", "12")
    _bfont = ("Arial", "12", "bold")
    _ufont = ("Arial", "12", "underline")

    def __init__(self, cells_x, cell_width, materials):
        super().__init__()
        self.cells_x = cells_x
        self.cell_width = cell_width
        self.materials = materials

        # Generate GUI Window
        self.title('ANTS - Mapper')
        self.minsize(400, 200)
        self._generate_buttons()
        self._generate_titles()
        self._generate_materials()
        
    def _generate_buttons(self):
        # buttonFont = font.Font(family='Arial', size=16, weight='bold')
        self.display_button = ttk.Button(self, text="Display")
        self.display_button["command"] = self._generate_display
        self.display_button.grid(row=0, column=0)
        self.save_button = ttk.Button(self, text="Save")
        self.save_button["command"] = self._save_object
        self.save_button.grid(row=0, column=1)
    
    def _generate_titles(self):
        cells_label = ttk.Label(self, font=self._bfont, \
                    text="Spatial Cells: {}".format(self.cells_x))
        cells_label.grid(row=1, column=0, padx=(10, 10))
        cell_width_label = ttk.Label(self, font=self._bfont, \
                    text="Cell Width: {}".format(self.cell_width))
        cell_width_label.grid(row=1, column=1, padx=10, pady=5)
        
        material_title = ttk.Label(self, font=self._ufont, \
                    text="Material")
        material_title.grid(row=2, column=0, padx=10, pady=5)
        cell_title = ttk.Label(self, font=self._ufont, \
                    text="Cell Number Range")
        cell_title.grid(row=2, column=1, padx=10, pady=5)

    def _generate_materials(self):
        self.spatial_cells = []
        self.material_names = []
        for item, material in enumerate(self.materials):
            self.material_names.append(material)
            temp_name = material.replace("material-","").replace("-", " ")
            mat_label = ttk.Label(self, text=temp_name, font=self._font)
            mat_label.grid(row=item+3, column=0, padx=10, pady=5)
            cell_numbers = ttk.Entry(self, font=self._font)
            self.spatial_cells.append(cell_numbers)
            cell_numbers.grid(row=item+3, column=1, padx=10, pady=5)

    def _generate_display(self):
        self._generate_mapping()
        graph_obj = LatexGraph(self.cells_x, self.cell_width, self.material_key, self.map_x)
        title = simpledialog.askstring("LaTeX Path", "Input LaTeX Directory Name")
        graph_obj.map_to_latex(title, path="")

    def _generate_mapping(self):
        self.material_key = {}
        material_id = 0
        self.map_x = np.ones((self.cells_x)) * -1
        for name, cells in zip(self.material_names, self.spatial_cells):
            self.material_key[name] = material_id
            cell_locs = self._format_range(cells.get())
            if not np.all(self.map_x[cell_locs] == -1):
                messagebox.showerror(title="Error!", \
                        message="Trying to overwrite allocated cell")
                return -1
            self.map_x[cell_locs] = material_id
            material_id += 1

    def _save_object(self):
        self._generate_mapping()
        answer = True
        if np.any(self.map_x == -1): 
            answer = messagebox.askokcancel(title="Warning!", \
                icon=messagebox.WARNING, \
                message="Not All Spatial Cells are Filled. Proceed?")
        if answer:
            file = filedialog.asksaveasfile(defaultextension="*.mpr", \
                title="Save File", filetypes=[("Mapper Files", "*.mpr")])
            if file:
                map_obj = Mapper(self.cells_x, self.cell_width, self.map_x, \
                             self.material_key)
                map_obj.save_map(file.name)
                self.quit()
        
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


class LatexGraph:
    _tikz_colors = ["red", "blue", "green", "magenta", "cyan", "yellow", \
                "black", "white"]

    def __init__(self, cells_x, cell_width, material_key, map_x):
        self.cells_x = cells_x
        self.cell_width = cell_width
        self.material_key = material_key
        self.map_x = map_x

    def _latex_compile(string, title, path):
        path = os.path.join(path, title)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + "/layout.tex", "w") as f:
            f.write(string)
        os.system("pdflatex -halt-on-error -output-directory {} layout.tex \
                            | grep '^!.*' -A200 --color=always".format(path))
        LatexGraph._latex_display(path)

    def _latex_display(path):
        if platform.system() == "Linux":
            os.system("xdg-open {}/layout.pdf".format(path))
        elif platform.system() == "Windows":
            os.system("cmd.exe /C start {}/layout.pdf".format(path))

    def _latex_header(self, title):
        string = r"\documentclass{{standalone}}{n}\usepackage{{tikz}}{n}".format(n="\n")
        string += r"\begin{{document}}{n}\begin{{tikzpicture}}".format(n="\n")
        string += r"[xscale={},yscale=0.025]{}".format(10 / self.cells_x, "\n")
        string += r"\node[align=center] at ({},60) {{{}}};".format( \
                                                0.5 * self.cells_x, title)
        string += r"\draw (-10,0) -- (0,0) node[align=center,below]{\small 0 cm};"
        string += r"\draw ({}, 0) -- ({}, 0);".format(self.cells_x, self.cells_x + 10)
        return string

    def _latex_materials(self, string):
        material_widths = [(x[0], len(list(x[1]))) for x in \
                                         itertools.groupby(self.map_x)]
        inv_map = LatexGraph.reformat_key(self.material_key)
        start_location = 0
        label_location = -30
        for mat, width in material_widths:
            string += r"\draw[draw=black,fill={},opacity=0.5]".format( \
                                            self._tikz_colors[int(mat)])
            string += r"({},0) rectangle ++ ({}, 50);".format( \
                                                start_location, width)
            try:
                _ = inv_map[mat]
                string += r"\draw[draw=black,fill={},opacity=0.5]".format( \
                                            self._tikz_colors[int(mat)])
                string += r"(0, {}) rectangle ++ (10, 10);".format(label_location)
                string += r"\node[align=center] at (25,{}) {{{}}};".format( \
                            label_location + 5, inv_map[mat])
                label_location -= 20
                del inv_map[mat]
            except KeyError:
                pass
            start_location += int(width)
            string += r"\node[align=center, below] at ({}, 0){{\small {}}};".format(\
                        start_location, self.cell_width * start_location)
        string = string.replace(";", ";\n")
        return string

    def map_to_latex(self, title="Untitled", path=""):
        string = self._latex_header(title)
        string = self._latex_materials(string)
        string += "\end{tikzpicture}\n\end{document}"
        LatexGraph._latex_compile(string, title, path)

    def reformat_key(dictionary):
        inv_map = {}
        for kk, vv in dictionary.items():
            kk = kk.replace("material-", "").replace("%", "\%")
            inv_map[vv] = kk.capitalize().strip()
        return inv_map

def gui(cells_x, cell_width, materials):
    if isinstance(materials, list):
        if "material-" not in materials[0]:
            materials = ["material-" + material for material in materials]
    window = MapperGUI(cells_x, cell_width, materials)
    window.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--materials", action="store", dest="materials", \
                    nargs="+", help="Materials to create a new mpr file")
    parser.add_argument("--cells", action="store", dest="cells", type=int, \
                    help="Default number of spatial cells")
    parser.add_argument("--width", action="store", dest="width", type=float, \
                        help="The width of one spatial cell")
    args = parser.parse_args()
    # materials = ["material-vacuum-reed"]
    materials = ["material-" + material for material in args.materials]
    window = MapperGUI(args.cells, args.width, materials)
    window.mainloop()
