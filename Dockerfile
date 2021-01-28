FROM pytorch/pytorch:latest

RUN apt-get update; apt-get install

COPY requirements.txt ./

RUN pip3 install -r requirements.txt

RUN python3 -m spacy download en

WORKDIR /appl/