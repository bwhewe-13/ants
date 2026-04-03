########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
########################################################################

import platform
import numpy as np
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup

Options.warning_errors = True

if platform.system() == "Darwin":
    ext = Extension("ants/*", sources=["ants/*.pyx"], include_dirs=[np.get_include()])
else:
    ext = Extension(
        "ants/*",
        sources=["ants/*.pyx"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
        include_dirs=[np.get_include()],
    )

setup(ext_modules=cythonize(ext, language_level="3"))
