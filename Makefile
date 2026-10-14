PYTHON ?= .venv/bin/python
DOCKER_API_VERSION ?= 1.43

.PHONY: setup data eda train lint test serve docker-build validate

setup:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/download_data.py

eda:
	$(PYTHON) scripts/run_eda.py

train:
	$(PYTHON) scripts/train.py

lint:
	.venv/bin/flake8 src scripts tests

test:
	$(PYTHON) -m pytest -q

serve:
	.venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000

docker-build:
	DOCKER_API_VERSION=$(DOCKER_API_VERSION) \
		docker build -t heart-disease-mlops:latest .

validate: lint test train docker-build
