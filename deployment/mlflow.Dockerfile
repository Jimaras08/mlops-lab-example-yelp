FROM python:3.7

RUN pip install PyMySQL==1.0.2 && \
    pip install psycopg2-binary==2.8.6 && \
    pip install mlflow[extras]==1.13.1

ENTRYPOINT ["mlflow", "server"]
