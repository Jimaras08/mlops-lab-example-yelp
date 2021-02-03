MAKEFILE_PATH = $(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_DIR = $(dir $(MAKEFILE_PATH))

IMAGE_LOCAL = model_server:v1.0
IMAGE_GCR = gcr.io/mlops-lab1-team3/$(IMAGE_LOCAL)

K8S_NS = mlflow-model

GIT_REPO = https://github.com/dmangonakis/Yelp_Dataset.git
GIT_BRANCH = $(shell git branch --show-current)


.PHONY: pass
pass:
	true

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
	kubectl create namespace $(K8S_NS) || true


.PHONY: k8s-undeploy
k8s-undeploy:
	kubectl -n $(K8S_NS) delete -f k8s/deployment.yaml

.PHONY: k8s-deploy
k8s-deploy: require-GIT_REPO require-GIT_BRANCH
k8s-deploy: require-IMAGE_GCR
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