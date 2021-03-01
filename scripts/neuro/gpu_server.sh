#!/bin/env bash
set -o xtrace

# cd <project-root>
[ -d "./src" ] || { echo "Must run from the project root!"; exit 1  ;}

echo "Uploading the project"
neuro cp -ru . -T storage:yelp_dataset --exclude data --exclude lightning_logs --exclude mlruns --exclude .git

NAME=yelp-server
neuro run \
  --name ${NAME} \
  --preset gpu-small-p \
  --volume storage:yelp_dataset:/project:rw \
  --volume secret:bucket-sa-key:/opt/developers-key.json \
  --env PYTHONPATH=/project \
  --env GOOGLE_APPLICATION_CREDENTIALS=/opt/developers-key.json \
  --env MLFLOW_TRACKING_URI=http://mlflow.lab1-team3.mlops.neu.ro:5000 \
  --env GIT_PYTHON_REFRESH=quiet \
  --detach \
  gcr.io/mlops-lab1-team3/yelp-dataset/model:v1.0 \
  bash
echo
echo "Server is running, please DO NOT FORGET TO KILL IT: 'neuro kill ${NAME}'"
echo
neuro attach ${NAME}
