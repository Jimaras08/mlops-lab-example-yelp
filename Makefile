IMAGE_LOCAL = yelp-dataset/model:v1.0
IMAGE_GCR = gcr.io/mlops-lab1-team3/$(IMAGE_LOCAL)


.PHONY: pass
pass:
	true

## Prepare remote training image
#

.PHONY: build-image
build-image:
	docker build . -t $(IMAGE_LOCAL)

.PHONY: push-image
push-image:
	gcloud auth configure-docker
	docker tag $(IMAGE_LOCAL) $(IMAGE_GCR)
	docker push $(IMAGE_GCR)

## Run training on Neu.ro platform
#

.PHONY: neuro-setup
neuro-setup:
	sh ./scripts/neuro/00_setup.sh

.PHONY: neuro-train
neuro-train:
	sh ./scripts/neuro/gpu_train.sh

.PHONY: neuro-server
neuro-server:
	sh ./scripts/neuro/gpu_server.sh

.PHONY: neuro-jupyter
neuro-jupyter:
	sh ./scripts/neuro/gpu_jupyter.sh
