# MLFlow Deployment in K8S working in GCP

## Setup
1. Managed Kubernetes Engine service is set up in Google Cloud
2. Managed PostgreSQL service is set up in Google Cloud (using [private IP address](https://cloud.google.com/sql/docs/postgres/connect-kubernetes-engine#private-ip) and authenticated via password)
3. Local tools: `gsutil` to access GCP, `helm`, `kubectl` to deploy and manage K8S cluster, `docker` to build images.

```sh
helm version  
# version.BuildInfo{Version:"v3.4.2", GitCommit:"23dd3af5e19a02d4f4baa5b2f242645a1a3af629", GitTreeState:"clean", GoVersion:"go1.15.6"}

kubectl version
# Client Version: version.Info{Major:"1", Minor:"20", GitVersion:"v1.20.1", GitCommit:"c4d752765b3bbac2237bf87cf0b1c2e307844666", GitTreeState:"clean", # BuildDate:"2020-12-18T12:09:25Z", GoVersion:"go1.15.5", Compiler:"gc", Platform:"linux/amd64"}
# Server Version: version.Info{Major:"1", Minor:"16+", GitVersion:"v1.16.15-gke.6000", GitCommit:"b02f5ea6726390a4b19d06fa9022981750af2bbc", GitTreeState:"clean", # BuildDate:"2020-11-18T09:16:22Z", GoVersion:"go1.13.15b4", Compiler:"gc", Platform:"linux/amd64

gsutil --version
# gsutil version: 4.57

docker --version
# Docker version 20.10.1, build 831ebeae96
```

## Build and push docker image

```sh
# authenticate local docker
gcloud auth configure-docker

REGISTRY=gcr.io/mlops-lab1-team3
IMAGE_NAME=mlflow-postgres
IMAGE_TAG=1.13.1

LOCAL=$IMAGE_NAME:$IMAGE_TAG
REMOTE=$REGISTRY/$LOCAL

# build docker image locally
docker build . -t $LOCAL

# push local image to GCR
docker tag $LOCAL $REMOTE
docker push $REMOTE

# verify the image was pushed
gcloud container images list --repository gcr.io/mlops-lab1-team3
gcloud container images list-tags $REGISTRY/$IMAGE_NAME
```

# Prepare the Cloud
1. Set up managed Kubernetes Engine service (GKE)
```sh
export GOOGLE_PROJECT="mlops-lab1-team3"
```

2. Set up managed PostgreSQL service (SQL)
- [GCP console](https://console.cloud.google.com/sql/instances/mlops-lab1-team3-mlflow-postgres/overview?authuser=1&project=mlops-lab1-team3)
- use internal IP
- create password-protected postgresql user `mlflow`
```sh
export POSTGRES_USER=mlflow
export POSTGRES_PASSWORD=<set password in GCP>
export POSTGRES_HOST=<see internal IP in GCP>
export POSTGRES_PORT=5432
```

3. Set up managed Google Cloud Storage service (GCS)

- create a bucket:
```sh
export BUCKET=gs://mlops-lab1-team3-mlflow
gsutil mb ${BUCKET}
```

- create Google Service Account with `write` permissions on objects within the bucket:
```sh
export GOOGLE_SA=developers
export GOOGLE_SA_FULL=${GOOGLE_SA}@${GOOGLE_PROJECT}.iam.gserviceaccount.com

# create service account
gcloud iam service-accounts create ${GOOGLE_SA} \
    --description="MLOps Lab 1, Team 3: Developers" \
    --display-name="mlops-lab1-team3-developers"

# grant permissions to this service account
gsutil iam ch serviceAccount:${GOOGLE_SA_FULL}:objectCreator ${BUCKET}

# download secret key to '/tmp/developers-key.json' and distribute it to developers
gcloud iam service-accounts keys create /tmp/${GOOGLE_SA}-key.json \
    --iam-account=${GOOGLE_SA_FULL}
```

- publish bucket on `read`: 
```sh
gsutil iam ch allUsers:objectViewer ${BUCKET}
```


# Deploy MLFlow

```sh
helm repo add larribas https://larribas.me/helm-charts
helm repo update

helm install \
    -n mlflow --create-namespace \
    mlflow larribas/mlflow \
    --set backendStore.postgres.username=${POSTGRES_USER} \
    --set backendStore.postgres.password=${POSTGRES_PASSWORD} \
    --set backendStore.postgres.host=${POSTGRES_HOST} \
    --set backendStore.postgres.port=${POSTGRES_PORT} \
    --set backendStore.postgres.database=mlflow \
    --set prometheus.expose=yes \
    --set defaultArtifactRoot=${BUCKET} \
    --set image.repository=larribas/mlflow \
    --set image.tag=1.9.1 \
    --set service.type=LoadBalancer
# (after a while)
# kubectl -n mlflow get all                                                                               
# NAME                          READY   STATUS    RESTARTS   AGE
# pod/mlflow-7686588c6b-p9jlm   1/1     Running   0          3m29s
# 
# NAME             TYPE           CLUSTER-IP     EXTERNAL-IP     PORT(S)          AGE
# service/mlflow   LoadBalancer   10.104.0.146   34.91.123.207   5000:31243/TCP   3m29s
# 
# NAME                     READY   UP-TO-DATE   AVAILABLE   AGE
# deployment.apps/mlflow   1/1     1            1           3m29s
# 
# NAME                                DESIRED   CURRENT   READY   AGE
# replicaset.apps/mlflow-7686588c6b   1         1         1       3m29s
```
Note `EXTERNAL-IP`: to access the MLFlow service, use `http://34.91.123.207:5000`


# Uninstall MLFlow Deployment
```sh
helm -n mlflow uninstall mlflow
```