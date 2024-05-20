########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \ 
#                     / ___ |/ /|  / / /  ___/ / 
#                    /_/  |_/_/ |_/ /_/  /____/  
#
########################################################################

from distutils.core import Extension, setup
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import setup, find_packages
import platform
import numpy as np

Options.warning_errors = True

if platform.system() == "Darwin":
    ext = Extension("ants/*", sources=["ants/*.pyx"],
                    include_dirs=[np.get_include()])
else:
    ext = Extension("ants/*", sources=["ants/*.pyx"], 
                    extra_compile_args=["-fopenmp"], 
                    extra_link_args=["-fopenmp"],
                    include_dirs=[np.get_include()])

setup(  
        name='ants',
        description=
        """A Neutron Transport Solution (ANTS) calculates the neutron 
        flux for both criticality and fixed source problems of one 
        dimensional slabs and spheres and two dimensional slabs
        using the discrete ordinates method. It looks to combine machine
        learning with collision based hybrid methods and speedup through
        Cython.""",
        version='1.0',
        author='Ben Whewell',
        author_email='ben.whewell@pm.me',
        url='https://github.com',

        packages=find_packages(),
        ext_modules=cythonize(ext, language_level="3"),
        include_package_data=True,
        install_requires=[
            "numpy", 
            "tqdm", 
            "cython"
        ], 
        extras_requires={
            "test": ["pytest"], 
            "examples": ["matplotlib", "seaborn", "h5py"]
        }
)
