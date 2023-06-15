
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def pytest_addoption(parser):
    parser.addoption("--mg1d", action="store_true", default=False, \
                     help="Runs one-dimensional multigroup problems if True")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--mg1d"):
        # --mg1d given in cli: do not skip multigroup tests
        return
    multigroup1d = pytest.mark.skip(reason="Run on --mg1d option")
    for item in items:
        if "multigroup1d" in item.keywords:
            item.add_marker(multigroup1d)