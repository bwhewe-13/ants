########################################################################
#                        ___    _   _____________
#                       /   |  / | / /_  __/ ___/
#                      / /| | /  |/ / / /  \__ \
#                     / ___ |/ /|  / / /  ___/ /
#                    /_/  |_/_/ |_/ /_/  /____/
#
# Runs the example notebooks in examples/ so API changes that break
# them are caught. The docs render the saved outputs without running
# the notebooks (nbsphinx_execute = "never"), so each notebook must
# also be saved with its outputs.
#
########################################################################

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))


@pytest.mark.notebooks
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_runs(path):
    notebook = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(EXAMPLES)}},
    )
    client.execute()


@pytest.mark.notebooks
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_saved_with_outputs(path):
    notebook = nbformat.read(path, as_version=4)
    not_run = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and cell.source.strip()
        and cell.execution_count is None
    ]
    assert not not_run, (
        f"{path.name} has code cells that were never run: {not_run}. "
        "Run the notebook and save it so the docs show its outputs."
    )
