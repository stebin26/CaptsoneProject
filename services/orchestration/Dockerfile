# syntax=docker/dockerfile:1

FROM apache/airflow:3.2.2-python3.12

ARG AIRFLOW_VERSION=3.2.2
ARG PYTHON_VERSION=3.12
ARG CONSTRAINTS_URL=https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        openjdk-17-jdk-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

USER airflow

COPY services/orchestration/requirements.txt /requirements.txt

RUN pip install --no-cache-dir \
        --constraint "${CONSTRAINTS_URL}" \
        -r /requirements.txt

COPY --chown=airflow:root packages/common /opt/ops/packages/common
COPY --chown=airflow:root services/ingestion /opt/ops/services/ingestion

RUN pip install --no-cache-dir \
        --constraint "${CONSTRAINTS_URL}" \
        /opt/ops/packages/common \
        /opt/ops/services/ingestion

RUN pip install --no-cache-dir pyspark==4.0.3