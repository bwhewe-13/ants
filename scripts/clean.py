"""Remove generated Cython sources, compiled extensions, and build folders.

Used by ``make clean``. Only ants/ and ants/utils/ are searched, so data
folders such as ants/sources/ are left alone.
"""

import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
FOLDERS = ("ants", "ants/utils")
PATTERNS = ("*.c", "*.cpp", "*.pyd", "*.so")
TREES = ("build", "ants.egg-info", "ants_sn.egg-info")

for folder in FOLDERS:
    for pattern in PATTERNS:
        for path in (ROOT / folder).glob(pattern):
            path.unlink()

for tree in TREES:
    shutil.rmtree(ROOT / tree, ignore_errors=True)
