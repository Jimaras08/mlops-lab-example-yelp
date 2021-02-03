MAKEFILE_PATH = $(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_DIR = $(dir $(MAKEFILE_PATH))

IMAGE_LOCAL = yelp-dataset/model-server:v1.0
IMAGE_GCR = gcr.io/mlops-lab1-team3/$(IMAGE_LOCAL)

K8S_NS = mlflow-model

GIT_BRANCH = $(shell git branch --show-current)


MLFLOW_MODEL_ARTIFACTS_PATH =

.PHONY: pass
pass:
	true

## Prepare remote app image
#

.PHONY: build-image
build-image:
	cd $(MAKEFILE_DIR) \
        && docker build . -t $(IMAGE_LOCAL)

.PHONY: push-image
push-image:
	gcloud auth configure-docker
	docker tag $(IMAGE_LOCAL) $(IMAGE_GCR)
	docker push $(IMAGE_GCR)

## Deploy the app to k8s
#

.PHONY: k8s-create-namespace
k8s-create-namespace:
	kubectl create namespace $(K8S_NS) || true


.PHONY: k8s-up-deployment
k8s-up-deployment: require-GIT_BRANCH require-MLFLOW_MODEL_RUN_ID
k8s-up-deployment:
	echo "Deploying from branch origin/$(GIT_BRANCH)"
	cd $(MAKEFILE_DIR) \
        && export GIT_BRANCH=$(GIT_BRANCH) \
        && export MLFLOW_MODEL_RUN_ID=$(MLFLOW_MODEL_RUN_ID) \
        && envsubst '$$GIT_BRANCH, $$MLFLOW_MODEL_RUN_ID' < k8s/deployment.yaml \
           | kubectl -n $(K8S_NS) apply -f -
	sleep 1
	kubectl -n $(K8S_NS) get all

.PHONY: k8s-down-deployment
k8s-down-deployment:
	kubectl -n $(K8S_NS) delete -f k8s/deployment.yaml


.PHONY: k8s-up-service
k8s-up-service:
	echo "Deploying from branch origin/$(GIT_BRANCH)"
	cd $(MAKEFILE_DIR) \
        && kubectl -n $(K8S_NS) apply -f k8s/service.yaml
	kubectl -n $(K8S_NS) get svc

.PHONY: k8s-down-service
k8s-down-service:
	kubectl -n $(K8S_NS) delete -f k8s/service.yaml


## utils
#

.SILENT: require-%
require-%:
	$(if $(value $(*)),,$(error Missing required argument $(*)))