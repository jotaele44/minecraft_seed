PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
export PYTHONPATH := $(CURDIR)

.PHONY: all fetch build validate preview run clean venv install

## Run the full pipeline (fetch → build → validate → preview)
all: fetch build validate preview

## Set up virtual environment
venv:
	python3 -m venv .venv

## Install dependencies into the venv
install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

## Download / generate the Puerto Rico DEM
fetch:
	$(PYTHON) tools/fetch_dem.py

## Re-fetch even if a valid DEM already exists
fetch-force:
	$(PYTHON) tools/fetch_dem.py --force

## Convert DEM to 8-bit grayscale PNG heightmap
build:
	$(PYTHON) tools/build_heightmap.py

## Convert DEM to 16-bit grayscale PNG heightmap
build-16:
	$(PYTHON) tools/build_heightmap.py --bits 16

## Validate all pipeline outputs
validate:
	$(PYTHON) tools/validate_outputs.py

## Generate terrain colourmap preview
preview:
	$(PYTHON) tools/preview_heightmap.py

## Run the full pipeline via orchestration script
run:
	$(PYTHON) tools/run_pipeline.py

## Run tests
test:
	$(PYTHON) -m pytest tests/ -v

## Remove generated outputs (keeps raw DEM)
clean:
	rm -f output/heightmap/puerto_rico_heightmap.png
	rm -f output/heightmap/puerto_rico_heightmap_16bit.png
	rm -f output/heightmap/puerto_rico_preview.png
	rm -f output/heightmap/heightmap_metadata.json
	rm -f output/worldpainter/import_settings.txt
	rm -f output/logs/source_audit.txt

## Remove everything including the raw DEM
clean-all: clean
	rm -f data/raw/puerto_rico_official_dem.tif
