#!/bin/env bash
set -o xtrace

# cd <project-root>
[ -d "./src" ] || { echo "Must run from the project root!"; exit 1  ;}

echo "Uploading the project"
neuro cp -ru . -T storage:yelp_dataset

NAME=yelp-jupyter
neuro run \
  --name ${NAME} \
  --preset gpu-small-p \
  --life-span 8h \
  --http 8888 \
  --volume storage:yelp_dataset:/project:rw \
  --volume secret:bucket-sa-key:/opt/developers-key.json \
  --env GOOGLE_APPLICATION_CREDENTIALS=/opt/developers-key.json \
  --env MLFLOW_TRACKING_URI=http://mlflow.lab1-team3.mlops.neu.ro:5000 \
  --env GIT_PYTHON_REFRESH=quiet \
  --env PYTHONPATH=/project \
  --detach \
  --life-span 8h \
  gcr.io/mlops-lab1-team3/yelp-dataset/model:v1.0 \
  bash -euo pipefail -c '
    pip install -Uq jupyter
    jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir=/project
  '
echo
echo "Jupyter is running, please DO NOT FORGET TO KILL IT: 'neuro kill ${NAME}'"
echo
neuro logs ${NAME}
