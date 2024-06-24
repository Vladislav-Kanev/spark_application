from pyspark.sql import SparkSession
import time
import argparse
from tqdm import tqdm
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from data_preprocess import preprocess_data
from utils import draw_graph, get_spark_memory_usage, mean


parser = argparse.ArgumentParser(description="Spark Application parameters")
parser.add_argument("--spark-url", type=str, required=True, help="URL for Spark Master")
parser.add_argument("--model-path", type=str, required=True, help="HDFS path to model")
parser.add_argument(
    "--data-path", type=str, required=True, help="HDFS path to CSV file"
)
parser.add_argument(
    "--optimized",
    action="store_true",
    help="Flag to use optimized version of the spark app",
)

args = parser.parse_args()


def run(df, model_path):
    model = RandomForestClassificationModel.load(model_path)

    df = preprocess_data(df)
    predictions = model.transform(df)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="isFraud", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    return accuracy


if __name__ == "__main__":
    spark = (
        SparkSession.builder.appName("SparkApp").master(args.spark_url).getOrCreate()
    )

    sc = spark.sparkContext
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)

    total_time = []
    total_RAM = []

    for i in tqdm(range(100)):
        start_time = time.time()

        df = spark.read.csv(args.data_path, header=True)

        if args.optimized:
            df.cache()
            df = df.repartition(4)
            accuracy = run(df, args.model_path)
        else:

            accuracy = run(df, args.model_path)

        end_time = time.time()

        total_time.append(end_time - start_time)
        total_RAM.append(get_spark_memory_usage(sc))

    draw_graph(total_time, total_RAM, "./result.png")

    print("Average memory(MB):", mean(total_RAM))
    print("Average time(c):", mean(total_time))
    spark.stop()
