import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.ticker as mticker
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import spark_partition_id
from pr3d.de import ConditionalGammaEVM, ConditionalGaussianMM, ConditionalGammaMixtureEVM
from pyspark.sql.types import StructType, StructField, FloatType
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


# open the ground truth written in the csv files
project_path = 'projects/qlen_benchmark/'
condition_labels = ['queue_length']
key_label = 'end2end_delay'
conditions = {
    '0_results':0, 
    '1_results':1, 
    '2_results':2, 
    '3_results':3, 
    '4_results':4, 
    '5_results':5,
    '6_results':6, 
    '7_results':7, 
    '8_results':8, 
    '9_results':9, 
    '10_results':10, 
    '11_results':11,
    '12_results':12, 
    '13_results':13, 
    '14_results':14,
}


model_addr = project_path + '/predictors/' + 'gmm_model_0.h5' # 'gmevm_model_0.h5'
# load the non conditional predictor
pr_model = ConditionalGaussianMM( #ConditionalGaussianMM, ConditionalGammaMixtureEVM
    h5_addr = model_addr,
)
seed = 12345
batch_size = 30000
logger.info(f"Predictor: {Path(model_addr).stem} is loaded.")

# plot axis
y_points = np.linspace(start=0,stop=250,num=100)

# figure 1
nrows = 3
ncols = 5
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols,5*nrows))
axes = axes.flat

for idx,records_dir in enumerate(conditions):
    records_path = project_path + 'records/' + records_dir
    ax = axes[idx]
    cond = conditions[records_dir]

    # open the empirical dataset
    all_files = os.listdir(records_path)
    files = []
    for f in all_files:
        if f.endswith(".parquet"):
            files.append(records_path + '/' + f)

    cond_df=spark.read.parquet(*files)
    total_count = cond_df.count()
    logger.info(f"Project path {records_path} parquet files are loaded.")
    logger.info(f"Total number of samples in this empirical dataset: {total_count}")

    emp_cdf = list()
    for y in y_points:
        delay_budget = y
        new_cond_df = cond_df \
            .where( cond_df[key_label] <= delay_budget )
        success_count = new_cond_df.count()
        emp_success_prob = success_count/total_count
        emp_cdf.append(emp_success_prob)

    ax.plot(
        y_points,
        emp_cdf,
        marker='.',
        label='simulation',
    )

    x = np.ones(len(y_points))*cond
    x = np.expand_dims(x, axis=1)
    
    y = np.array(y_points, dtype=np.float64)
    y = y.clip(min=0.00)
    prob, logprob, pred_cdf = pr_model.prob_batch(x, y)

    ax.plot(
        y_points,
        pred_cdf,
        marker='.',
        label='prediction',
    )

# figure 2
fig.tight_layout()
plt.savefig(project_path+'qlen_validation.png')