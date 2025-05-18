VENV      := .venv
PYTHON    := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
UVICORN   := $(VENV)/bin/uvicorn
PYTEST    := $(VENV)/bin/pytest

.PHONY: venv install activate run test clean

## Create virtual environment
venv:
	@echo "Creating virtual environment in $(VENV)..."
	python3 -m venv $(VENV)

## Install requirements into venv
install: venv
	@echo "Upgrading pip and installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

## Show how to activate the venv
activate:
	@echo "Run the following to activate your venv:"
	@echo "  source $(VENV)/bin/activate"

## Start FastAPI server (requires install)
run: install
	@echo "Starting server on http://0.0.0.0:8000 ..."
	$(UVICORN) main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload

## Run pytest against our test suite
test: install
	@echo "Running API tests..."
	$(PYTEST) --maxfail=1 --disable-warnings -q

## Remove venv
clean:
	@echo "Removing virtual environment..."
	rm -rf $(VENV)