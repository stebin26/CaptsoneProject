# syntax=docker/dockerfile:1

FROM apache/spark:4.0.3

ARG PG_JDBC_VERSION=42.7.4

USER root

ADD --chmod=644 \
    https://repo1.maven.org/maven2/org/postgresql/postgresql/${PG_JDBC_VERSION}/postgresql-${PG_JDBC_VERSION}.jar \
    /opt/spark/jars/postgresql-${PG_JDBC_VERSION}.jar

RUN pip install --no-cache-dir psycopg2-binary

USER spark