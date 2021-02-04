FROM python:3.8-slim-buster

EXPOSE 8501
#EXPOSE 5432

WORKDIR /source/app

COPY . .

RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["streamlit", "run"]

#WORKDIR /appl/
