IMAGE_LOCAL = yelp-dataset/model-operator:v1.1
IMAGE_GCR = gcr.io/mlops-lab1-team3/$(IMAGE_LOCAL)

K8S_NS = mlflow-model

GIT_BRANCH = $(shell git branch --show-current)


.PHONY: pass
pass:
	true

## Prepare remote app image
#

.PHONY: build-image
build-image:
	docker build . -t $(IMAGE_LOCAL)

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
k8s-up-deployment: require-GIT_BRANCH
k8s-up-deployment:
	echo "Deploying from branch origin/$(GIT_BRANCH)"
	kubectl -n $(K8S_NS) apply -f k8s/service-account.yaml \
        && export GIT_BRANCH=$(GIT_BRANCH) \
        && envsubst '$$GIT_BRANCH' < k8s/deployment.yaml | kubectl -n $(K8S_NS) apply -f -
	sleep 1
	kubectl -n $(K8S_NS) get all

.PHONY: k8s-down-deployment
k8s-down-deployment:
	kubectl -n $(K8S_NS) delete -f k8s/deployment.yaml
	kubectl -n $(K8S_NS) delete -f k8s/service-account.yaml


.PHONY: k8s-up-service
k8s-up-service:
	echo "Deploying from branch origin/$(GIT_BRANCH)"
	kubectl -n $(K8S_NS) apply -f k8s/service.yaml
	kubectl -n $(K8S_NS) get svc

.PHONY: k8s-down-service
k8s-down-service:
	kubectl -n $(K8S_NS) delete -f k8s/service.yaml


## utils
#

.SILENT: require-%
require-%:
	$(if $(value $(*)),,$(error Missing required argument $(*)))