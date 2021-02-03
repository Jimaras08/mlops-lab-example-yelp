FROM gcr.io/mlops-lab1-team3/yelp-dataset/model:v1.0

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt /opt/requirements.txt

RUN pip install -U --no-cache pip \
    && pip install -U --no-cache mlflow==1.13.1 google-cloud-storage==1.35.0 \
    && pip install -r /opt/requirements.txt
