import os

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", 5432))
POSTGRES_USER = os.environ.get("POSTGRES_USER", "mlflow-inference")
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
POSTGRES_DATABASE = os.environ.get("POSTGRES_DATABASE", "mlflow-inference")
POSTGRES_TABLE = "predicts"
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DATABASE}"

MODEL_HOST = os.environ["MODEL_HOST"]
MODEL_PORT = int(os.environ["MODEL_PORT"])
MODEL_URI = f"http://{MODEL_HOST}:{MODEL_PORT}"
