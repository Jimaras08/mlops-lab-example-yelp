# Example of working with MLFlow

This is example from the [official MLFlow tutorial](https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html) adapted to work with our deployed MLFlow server.

First, set up your environment:
```sh
# path to downloaded key for Google Service Account (needed to 
# push model artifacts to the remote object storage):
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/developers-key.json
# path to external IP of MLFlow service:
export MLFLOW_TRACKING_URI=http://34.90.15.80:5000
```

Now, train the model:
```sh
# Train model in local python environment:
python train.py 0.5                                  
#Elasticnet model (alpha=0.500000, l1_ratio=0.500000):
#  RMSE: 0.82224284975954
#  MAE: 0.6278761410160693
#  R2: 0.12678721972772689

python train.py 0.7 0.6
#Elasticnet model (alpha=0.700000, l1_ratio=0.600000):
#  RMSE: 0.8591618564339473
#  MAE: 0.6483506286491303
#  R2: 0.04661163452161077

# Alternatively, train model in a local anaconda environment:
mlflow run . -Palpha=0.5
```
