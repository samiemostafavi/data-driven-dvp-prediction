import os
import pandas as pd
from pathlib import Path
import json
from deepdiff import DeepDiff
from functools import reduce
import operator
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.ticker as mticker
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import spark_partition_id
from pyspark.sql import Window
from pr3d.de import ConditionalGammaEVM
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.sql.types import StructType,StructField, FloatType
from petastorm import TransformSpec
from os.path import dirname, abspath
import tensorflow as tf
import warnings
from loguru import logger

warnings.filterwarnings("ignore")

def init_spark():

    # "spark.driver.memory" must not exceed the total memory of the device: SWAP + RAM
    # "spark.sql.execution.arrow.pyspark.enabled" is for faster conversion of Pandas dataframe to spark

    spark = SparkSession.builder \
        .master("local") \
        .appName("LoadParquets") \
        .config("spark.executor.memory","6g") \
        .config("spark.driver.memory", "70g") \
        .config("spark.driver.maxResultSize",0) \
        .getOrCreate()

    sc = spark.sparkContext
    return spark,sc

# init Spark
spark,sc = init_spark()

# Set a cache directory on DBFS FUSE for intermediate data.
file_path = dirname(abspath(__file__))
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, 'file://' + file_path + '/sparkcache')
logger.info(f"Spark cache folder is set up at: {'file://' + file_path + '/sparkcache'}")

# open the ground truth written in the csv files
project_folder = "projects/tail_benchmark/"
project_paths = [project_folder+name for name in os.listdir(project_folder) if os.path.isdir(os.path.join(project_folder, name))]
#project_paths = ['projects/tail_benchmark/p1_results']

condition_labels = ['queue_length', 'longer_delay_prob']
key_label = 'end2end_delay'

# Predictors benchmarking
# open the dataframe from parquet files
for project_path in project_paths:
    records_path = project_path + '/records/'
    all_files = os.listdir(records_path)
    files = []
    for f in all_files:
        if f.endswith(".parquet"):
            files.append(records_path + f)

    df=spark.read.parquet(*files)
    df = df.select(*condition_labels)

    logger.info(f"The number of samples in the dataset: {df.count()}")

    # open predictors
    records_path = project_path + '/predictors/'
    all_files = os.listdir(records_path)
    files = []
    for f in all_files:
        if f.endswith(".h5"):
            files.append(records_path + f)

    # limit the predictor
    files = [files[0]]

    records_pred_path= project_path + '/records_predicted/'
    os.makedirs(records_path, exist_ok=True)
    for idx,model_addr in enumerate(files):
        logger.info(f"Working on {Path(model_addr).stem}")

        # draw predictions
        # load the non conditional predictor
        pr_model = ConditionalGammaEVM(
            h5_addr = model_addr,
        )
        rng = tf.random.Generator.from_seed(12345)
        batch_size = 1000000

        # get pyspark dataframe in batches [https://stackoverflow.com/questions/60645256/how-do-you-get-batches-of-rows-from-spark-using-pyspark]
        count = df.count() 
        partitions = int(count/batch_size)
        df = df.repartition(partitions)
        # add the partition_number as a column
        df = df.withColumn('partition_num', spark_partition_id())
        df.persist()

        total_partition = [int(row.partition_num) for row in 
            df.select('partition_num').distinct().collect()]

        # create an empty pyspark dataset to append to it
        pred_df = spark.createDataFrame([], schema = StructType([
            StructField(label, FloatType(), True) for label in [*condition_labels, key_label]
        ]))

        i = 0
        for each_df in total_partition:
            logger.info(f"Producing samples for partition {i+1}/{len(total_partition)}")
            i = i+1
            x_input_pd = df.where(df.partition_num == each_df) \
                .select(*condition_labels) \
                .toPandas()

            x_input_dict = { label:x_input_pd[label].to_numpy() for label in condition_labels }

            pred_np = pr_model.sample_n(
                x_input_dict,
                rng = rng,
            )
            doc = spark.createDataFrame(pd.DataFrame({**x_input_dict, key_label:pred_np}))
            pred_df = pred_df.union(doc)

        print(pred_df.count())
        print(pred_df.show())

        pred_df.write.parquet(records_pred_path + Path(model_addr).stem + "_n.parquet")
    exit(0)