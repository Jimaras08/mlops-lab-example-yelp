FROM python:3.7

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt /opt/requirements.txt

RUN pip install -U pip \
    && pip install -r /opt/requirements.txt
