FROM maven:3-jdk-11 as builder
RUN apt -y update && apt -y install make \
&& git clone --depth 1 https://github.com/apache/cassandra-harry.git \
&& cd cassandra-harry \
&& make standalone

FROM openjdk:11 as app
COPY --from=builder /cassandra-harry /cassandra-harry
COPY external.yaml /cassandra-harry/conf/external.yaml
COPY logback-dtest.xml /cassandra-harry/test/conf/logback-dtest.xml

RUN sed -i '2s;^;HARRY_HOME=/cassandra-harry\n;'  /cassandra-harry/scripts/cassandra-harry
RUN sed -i 's/external-.*-SNAPSHOT.jar/external-*-SNAPSHOT.jar/' /cassandra-harry/scripts/cassandra-harry
RUN ln -svf /cassandra-harry/scripts/cassandra-harry /usr/bin/cassandra-harry
RUN ln -svf /dev/null /cassandra-harry/validation.log
RUN ln -svf /dev/null /cassandra-harry/operation.log

WORKDIR /cassandra-harry
