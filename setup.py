########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
########################################################################

import os
import platform
import warnings

import numpy as np
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup

Options.warning_errors = True


def _find_libomp():
    """Return the prefix of a libomp install, or None."""
    # Prefer conda's libomp: NumPy's OpenBLAS already loads it
    candidates = [
        os.environ.get("CONDA_PREFIX"),
        "/opt/homebrew/opt/libomp",  # Homebrew on Apple silicon
        "/usr/local/opt/libomp",  # Homebrew on Intel
    ]
    for prefix in filter(None, candidates):
        header = os.path.join(prefix, "include", "omp.h")
        library = os.path.join(prefix, "lib", "libomp.dylib")
        if os.path.isfile(header) and os.path.isfile(library):
            return prefix
    return None


if platform.system() == "Darwin":
    # Apple clang has no -fopenmp driver flag and ships no OpenMP runtime,
    # so the preprocessor flag is passed through and libomp linked directly.
    libomp = _find_libomp()
    if libomp is None:
        warnings.warn("libomp not found; building without OpenMP")
        compile_args, link_args = [], []
    else:
        lib_dir = os.path.join(libomp, "lib")
        compile_args = [
            "-Xpreprocessor",
            "-fopenmp",
            "-I" + os.path.join(libomp, "include"),
        ]
        link_args = ["-L" + lib_dir, "-lomp", "-Wl,-rpath," + lib_dir]
elif platform.system() == "Windows":
    # MSVC ignores -fopenmp (warning D9002), so Windows needs /openmp
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

ext_modules = cythonize([ext, ext_utils], language_level="3")

if os.environ.get("CIBUILDWHEEL") == "1":
    # The .pyx files ask for -march=native, which would tie the wheels to
    # the CPU of the build machine
    for module in ext_modules:
        module.extra_compile_args = [
            arg for arg in module.extra_compile_args if arg != "-march=native"
        ]

setup(ext_modules=ext_modules)
