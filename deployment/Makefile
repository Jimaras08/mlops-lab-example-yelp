IMAGE_LOCAL = yelp-dataset/mlflow:1.13.1
IMAGE_GCR = gcr.io/mlops-lab1-team3/$(IMAGE_LOCAL)


.PHONY: pass
pass:
	true

## Prepare remote mlflow image
#

.PHONY: build-mlflow-image
build-mlflow-image:
	docker build . -f mlflow.Dockerfile -t $(IMAGE_LOCAL)

.PHONY: push-mlflow-image
push-mlflow-image:
	gcloud auth configure-docker
	docker tag $(IMAGE_LOCAL) $(IMAGE_GCR)
	docker push $(IMAGE_GCR)
