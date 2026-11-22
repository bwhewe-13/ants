# Developer shortcuts for building, testing, and documenting ANTS.
# Works with GNU make on Linux, macOS, and Windows (PowerShell, cmd, or
# Git Bash). Recipes avoid quotes and shell commands so that every shell
# runs them the same way.
#
#   make build    compile the Cython extensions in place
#   make clean    remove generated C++, compiled extensions, and build/
#   make rebuild  clean then build
#   make install  clean then editable install (pip install -e .)
#   make test     run the test suite in parallel
#   make test-mg  run the test suite including the multigroup problems
#   make docs     build the HTML documentation
#
# Use another interpreter with: make build PYTHON=python3

PYTHON ?= python

.PHONY: build clean rebuild install test test-mg docs

build:
	$(PYTHON) setup.py build_ext --inplace

clean:
	$(PYTHON) scripts/clean.py

rebuild: clean build

install: clean
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest tests -n auto

test-mg:
	$(PYTHON) -m pytest tests -n auto --mg1d --mg2d

docs:
	$(MAKE) -C docs html
