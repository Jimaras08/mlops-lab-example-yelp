FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN python -m spacy download en

WORKDIR /project/