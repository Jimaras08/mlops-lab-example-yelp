MAKEFILE_PATH = $(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_DIR = $(dir $(MAKEFILE_PATH))

IMAGE_LOCAL = model_proxy:v1.0
IMAGE_GCR = gcr.io/mlops-lab1-team3/$(IMAGE_LOCAL)

K8S_NS = mlflow-seldon

POSTGRES_HOST = 
POSTGRES_PASSWORD =
POSTGRES_USER = mlflow-inference

GIT_REPO = https://github.com/dmangonakis/Yelp_Dataset.git
GIT_BRANCH = $(shell git branch --show-current)

## Run the app
#

.PHONY: run-app
run-app:
	cd $(MAKEFILE_DIR) \
        && uvicorn app.main:app --host 0.0.0.0 --port 8080 --access-log --use-colors

## Prepare remote app image
#

.PHONY: build-image
build-image:
	cd $(MAKEFILE_DIR) \
        && docker build . -t $(IMAGE_LOCAL)

.PHONY: k8s-push-image
k8s-push-image:
	gcloud auth configure-docker
	docker tag $(IMAGE_LOCAL) $(IMAGE_GCR)
	docker push $(IMAGE_GCR)

## Deploy the app to k8s
#

.PHONY: k8s-create-namespace
k8s-create-namespace:
	kubectl create namespace $(K8S_NS)

.PHONY: k8s-create-postgres-secret
k8s-create-postgres-secret: require-POSTGRES_USER require-POSTGRES_PASSWORD
k8s-create-postgres-secret:
	kubectl -n $(K8S_NS) create secret generic postgres-config \
		--from-literal user=$(POSTGRES_USER) \
		--from-literal password=$(POSTGRES_PASSWORD)


.PHONY: k8s-undeploy
k8s-undeploy:
	kubectl -n mlflow-seldon delete -f k8s/deployment.yaml

.PHONY: k8s-deploy
k8s-deploy: require-GIT_REPO require-GIT_BRANCH
k8s-deploy: require-IMAGE_GCR require-POSTGRES_HOST
k8s-deploy:
	echo "Deploying from branch origin/$(GIT_BRANCH)"
	cd $(MAKEFILE_DIR) \
        && export GIT_REPO=$(GIT_REPO) \
        && export GIT_BRANCH=$(GIT_BRANCH) \
        && export IMAGE_GCR=$(IMAGE_GCR) \
        && export POSTGRES_HOST=$(POSTGRES_HOST) \
        && envsubst < k8s/deployment.yaml | kubectl -n $(K8S_NS) apply -f -
	kubectl -n $(K8S_NS) get all


.PHONY: k8s-unservice
k8s-unservice:
	kubectl -n mlflow-seldon delete -f k8s/service.yaml

.PHONY: k8s-service
k8s-service:
	echo "Deploying from branch origin/$(GIT_BRANCH)"
	cd $(MAKEFILE_DIR) \
        && kubectl -n $(K8S_NS) apply -f k8s/service.yaml
	kubectl -n $(K8S_NS) get svc


## utils
#

.SILENT: require-%
require-%:
	$(if $(value $(*)),,$(error Missing required argument $(*)))
