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
    ext_utils = Extension(
        "ants/utils/*", sources=["ants/utils/*.pyx"], include_dirs=[np.get_include()]
    )
else:
    # MSVC ignores -fopenmp (warning D9002), so Windows needs /openmp
    if platform.system() == "Windows":
        compile_args, link_args = ["/openmp"], []
    else:
        compile_args, link_args = ["-fopenmp"], ["-fopenmp"]
    ext = Extension(
        "ants/*",
        sources=["ants/*.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=[np.get_include()],
    )
    ext_utils = Extension(
        "ants/utils/*",
        sources=["ants/utils/*.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=[np.get_include()],
    )

setup(ext_modules=cythonize([ext, ext_utils], language_level="3"))
