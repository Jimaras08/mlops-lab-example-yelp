FROM python:3.7

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN curl -LO https://storage.googleapis.com/kubernetes-release/release/v1.20.1/bin/linux/amd64/kubectl \
    && chmod +x ./kubectl \
    && mv ./kubectl /usr/local/bin/kubectl \
    && kubectl version --client

# Install envsubst
RUN apt-get update \
    && apt-get -y install gettext-base

COPY requirements.txt /opt/requirements.txt

RUN pip install -U --no-cache pip \
    && pip install -r /opt/requirements.txt
