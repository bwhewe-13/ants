import pytest


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def pytest_addoption(parser):
    parser.addoption(
        "--mg1d",
        action="store_true",
        default=False,
        help="Runs one-dimensional multigroup problems if True",
    )
    parser.addoption(
        "--mg2d",
        action="store_true",
        default=False,
        help="Runs two-dimensional multigroup problems if True",
    )


def pytest_collection_modifyitems(config, items):
    # Skip each multigroup marker unless its own option is given, so that
    # --mg1d and --mg2d work independently
    for option, keyword in (("--mg1d", "multigroup1d"), ("--mg2d", "multigroup2d")):
        if config.getoption(option):
            continue
        skip = pytest.mark.skip(reason=f"Run on {option} option")
        for item in items:
            if keyword in item.keywords:
                item.add_marker(skip)
