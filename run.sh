#!/bin/bash

DOCKER_COMPOSE_FILE="hadoop/docker-compose-1-datanode.yml"

OPTIMIZED_FLAG=""
OUTPUT_FILE_NAME="not-optimized"

NAMENODE="namenode"
SPARK_MASTER="spark-master"
SPARK_WORKER="spark-worker-1"

if [ "$2" == "optimized" ]; then
  OPTIMIZED_FLAG="--optimized"
  OUTPUT_FILE_NAME="optimized"
fi

if [ "$1" == "3" ]; then
  DOCKER_COMPOSE_FILE="hadoop/docker-compose-3-datanode.yml"
  OUTPUT_FILE_NAME="$OUTPUT_FILE_NAME-$1"

  NAMENODE="namenode3"
  SPARK_MASTER="spark-master3"
  SPARK_WORKER="spark-worker-3"
fi

RES_FILE_NAME="result.png"
OUTPUT_FILE_NAME="$OUTPUT_FILE_NAME.png"


echo "Using Docker Compose file: $DOCKER_COMPOSE_FILE"
echo "Optimized flag is set to: $OPTIMIZED_FLAG"
echo "Namenode docker name: $NAMENODE"
echo "Spark master docker name: $SPARK_MASTER"
echo "Spark worker docker name: $SPARK_WORKER"

docker-compose -f $DOCKER_COMPOSE_FILE up -d --build

echo "Waiting for containers to start..."
sleep 15

docker cp dataset/PS_20174392719_1491204439457_log.csv $NAMENODE:/
docker cp model/random_forest_classifier $NAMENODE:/

docker exec -it $NAMENODE hdfs dfs -put PS_20174392719_1491204439457_log.csv /
docker exec -it $NAMENODE hdfs dfs -put random_forest_classifier /

docker exec -it $SPARK_WORKER rm -rf result.png

docker exec -it $SPARK_WORKER /spark/bin/spark-submit \
    /opt/spark-apps/spark.py \
    --spark-url "spark://$SPARK_MASTER:7077" \
    --model-path "hdfs://$NAMENODE:9001/random_forest_classifier" \
    --data-path "hdfs://$NAMENODE:9001/PS_20174392719_1491204439457_log.csv" \
    $OPTIMIZED_FLAG

docker cp $SPARK_WORKER:./$RES_FILE_NAME ./results/$OUTPUT_FILE_NAME