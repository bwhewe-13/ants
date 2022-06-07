from distutils.core import setup, Extension
from Cython.Build import cythonize

# ext = Extension(name="wrap_fib", source=["cfibc.c", "wrap_fib.pyx"])
# ext = ["hermite_splines.pyx", "source_iteration.pyx", "splines.pyx"] 
ext = ["multi_group.pyx", "x_sweeps.pyx"] #, "x_sweeps.pxd"] 

setup(ext_modules=cythonize(ext, language_level="3"))

