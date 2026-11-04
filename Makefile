# Makefile
SHELL := /bin/bash
.PHONY: venv

VENV ?= venv
PYTHON ?= python3

venv:
	@$(PYTHON) -m venv $(VENV) || { echo "venv creation failed (Debian/Ubuntu/WSL: sudo apt install python3-venv)"; exit 1; }
	$(VENV)/bin/pip install -q --upgrade pip
	@$(VENV)/bin/pip install .; rc=$$?; rm -rf build project1.egg-info; exit $$rc
