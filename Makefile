GCS_BUCKET = gs://mlops-lab1-team3-mlflow
LOCAL_DATASET ?= ~/.data

## Install Python Dependencies
requirements: test_environment
	python -m pip install -U pip setuptools wheel
	python -m pip install -r requirements.txt

## Make Dataset
data: requirements
	python src/data/make_dataset.py data/raw data/processed

download-dataset:
	gsutil cp $(GCS_BUCKET)/dataset_split.pkl $(LOCAL_DATASET)

