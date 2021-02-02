#!/bin/env bash
set -o xtrace

# cd <project-root>
[ -d "./src" ] || { echo "Must run from the project root!"; exit 1  ;}

NAME=yelp-server


neuro run \
  --name ${NAME} \
  --preset gpu-small-p \
  --volume storage:yelp_dataset:/project:rw \
  --env PYTHONPATH=/project \
  --env GOOGLE_APPLICATION_CREDENTIALS=/project/developers-key.json \
  --env MLFLOW_TRACKING_URI=http://mlflow.lab1-team3.neu.ro:5000 \
  --detach \
  image:yelp_dataset:v1.0 \
  bash
echo "======================================================================="
echo "Server is running, please don't forget to kill it: 'neuro kill ${NAME}'"
echo "======================================================================="
neuro attach ${NAME}
