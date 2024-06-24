import matplotlib.pyplot as plt
from pyspark import SparkContext


def get_spark_memory_usage(sparkContext: SparkContext):
    executor_memory_status = sparkContext._jsc.sc().getExecutorMemoryStatus()
    executor_memory_status_dict = (
        sparkContext._jvm.scala.collection.JavaConverters.mapAsJavaMapConverter(
            executor_memory_status
        ).asJava()
    )
    total_memory_used = 0

    for _, values in executor_memory_status_dict.items():
        total_memory = values._1()
        free_memory = values._2()
        used_memory = total_memory - free_memory
        total_memory_used += used_memory

    return total_memory_used / (1024 * 1024)


def mean(data):
    return sum(data) / len(data)


def draw_graph(total_time, total_RAM, path_name):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.hist(total_time, bins=20)
    plt.xlabel("Время(c)")
    plt.ylabel("Частота")
    plt.title("Затраченное время")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(total_RAM, bins=20)
    plt.xlabel("RAM(MB)")
    plt.ylabel("Частота")
    plt.title("RAM")
    plt.grid(True)

    plt.savefig(path_name)
