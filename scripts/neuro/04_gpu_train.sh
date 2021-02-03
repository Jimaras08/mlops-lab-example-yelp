#!/bin/env bash
set -o xtrace

# cd <project-root>
[ -d "./src" ] || { echo "Must run from the project root!"; exit 1  ;}

echo "Uploading the project"
neuro cp -ru . -T storage:yelp_dataset

NAME=yelp-train
neuro run \
  --name ${NAME} \
  --preset gpu-small-p \
  --volume storage:yelp_dataset:/project:rw \
  --volume secret:bucket-sa-key:/opt/developers-key.json \
  --env PYTHONPATH=/project \
  --env GOOGLE_APPLICATION_CREDENTIALS=/opt/developers-key.json \
  --env MLFLOW_TRACKING_URI=http://mlflow.lab1-team3.neu.ro:5000 \
  --env GIT_PYTHON_REFRESH=quiet \
  --detach \
  image:/artemyushkovskiy/yelp_dataset:v1.0 \
  mlflow run /project --no-conda -P max_epochs=15
echo
echo "Server is running, please DO NOT FORGET TO KILL IT: 'neuro kill ${NAME}'"
echo
neuro logs ${NAME}
