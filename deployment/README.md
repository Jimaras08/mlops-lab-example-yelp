- Download dataset:

`gsutil cp gs://mlops-lab1-team3-mlflow/dataset_split.pkl dataset_split.pkl`

- Build docker image:

`sudo docker build --no-cache -t yelp_dataset:v1.0 .`
`docker-compose up`