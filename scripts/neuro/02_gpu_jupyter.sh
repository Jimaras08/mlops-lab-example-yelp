#!/bin/env bash
set -o xtrace

# cd <project-root>
[ -d "./src" ] || { echo "Must run from the project root!"; exit 1  ;}

NAME=yelp-jupyter


neuro run \
  --name ${NAME} \
  --preset gpu-small-p \
  --life-span 8h \
  --http 8888 \
  --volume storage:yelp_dataset:/project:rw \
  --volume secret:bucket-sa-key:/opt/developers-key.json \
  --env GOOGLE_APPLICATION_CREDENTIALS=/opt/developers-key.json \
  --env MLFLOW_TRACKING_URI=http://mlflow.lab1-team3.neu.ro:5000 \
  --env PYTHONPATH=/project \
  --detach \
  image:yelp_dataset:v1.0 \
  bash -euo pipefail -o xtrace -c '
    pip install -Uq jupyter
    jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir=/project
  '
echo "========================================================================"
echo "Jupyter is running, please don't forget to kill it: 'neuro kill ${NAME}'"
echo "========================================================================"
neuro logs ${NAME}
