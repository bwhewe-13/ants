"""Copy the tests and pytest settings into the current folder.

Used by cibuildwheel. tests/ is a package, so pytest run from the source
tree puts the tree on sys.path and imports the uncompiled ants/ instead of
the installed wheel.
"""

import pathlib
import shutil
import sys

project = pathlib.Path(sys.argv[1])
shutil.copytree(project / "tests", "tests")
shutil.copy(project / "pyproject.toml", ".")
