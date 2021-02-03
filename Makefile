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

.PHONY: train
train:
	sh ./scripts/neuro/gpu_train.sh
