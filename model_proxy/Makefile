MAKEFILE_PATH = $(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_DIR = $(dir $(MAKEFILE_PATH))

IMAGE_LOCAL = yelp-dataset/model-proxy:v1.0
IMAGE_GCR = gcr.io/mlops-lab1-team3/$(IMAGE_LOCAL)

K8S_NS = mlflow-model

POSTGRES_HOST =
POSTGRES_PORT = 5432
POSTGRES_PASSWORD =
POSTGRES_USER =
POSTGRES_DATABASE =

MODEL_HOST =
MODEL_PORT =

GIT_BRANCH = $(shell git branch --show-current)


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

.PHONY: k8s-create-postgres-secret
k8s-create-postgres-secret: require-POSTGRES_USER require-POSTGRES_PASSWORD require-POSTGRES_DATABASE
k8s-create-postgres-secret:
	kubectl -n $(K8S_NS) create secret generic postgres-config \
		--from-literal user=$(POSTGRES_USER) \
		--from-literal password=$(POSTGRES_PASSWORD) \
		--from-literal database=$(POSTGRES_DATABASE)


.PHONY: k8s-up-deployment
k8s-up-deployment: require-GIT_BRANCH
k8s-up-deployment: require-POSTGRES_HOST require-POSTGRES_PORT
k8s-up-deployment: require-MODEL_HOST require-MODEL_PORT
k8s-up-deployment:
	echo "Deploying from branch origin/$(GIT_BRANCH)"
	cd $(MAKEFILE_DIR) \
        && export GIT_BRANCH=$(GIT_BRANCH) \
        && export POSTGRES_HOST=$(POSTGRES_HOST) \
        && export POSTGRES_PORT=$(POSTGRES_PORT) \
        && export MODEL_HOST=$(MODEL_HOST) \
        && export MODEL_PORT=$(MODEL_PORT) \
        && envsubst '$$GIT_BRANCH, $$POSTGRES_HOST, $$POSTGRES_PORT, $$MODEL_HOST, $$MODEL_PORT' \
            < k8s/deployment.yaml \
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
