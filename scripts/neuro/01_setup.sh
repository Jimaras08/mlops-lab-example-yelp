#!/bin/env bash
set -o xtrace

# cd <project-root>
[ -d "./src" ] || { echo "Must run from the project root!"; exit 1  ;}

# Parse arguments
for i in "$@"; do
  case $i in
    --bucket-sa-key-file=*)
    BUCKET_SA_KEY_FILE="${i#*=}"
    shift # past argument=value
    ;;
  esac
done
if [ ! "${BUCKET_SA_KEY_FILE}" ]; then
  echo "Missing required argument: --bucket-sa-key-file=..."
  exit 1
fi

# --

echo "Installing Neu.ro platform clients"
pip install -Uq neuro-cli neuro-extras

echo "Logging in to the Neu.ro platform"
neuro login  # register with email and get free 100h gpu
neuro config switch-cluster neuro-compute


echo "Securely uploading bucket credentials"
neuro secret add bucket-sa-key @${BUCKET_SA_KEY_FILE}

echo "Uploading training code"
neuro cp -ru . -T storage:yelp_dataset

echo "Building training image"
neuro-extras image build -f Dockerfile . image:yelp_dataset:v1.0
