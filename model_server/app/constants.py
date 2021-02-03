import os

SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = os.environ.get("SERVER_PORT", 8080)
SERVER_NUM_WORKERS = os.environ.get("SERVER_NUM_WORKERS", 1)

MODEL_ARTIFACTS_PATH = os.environ["MODEL_ARTIFACTS_PATH"]
