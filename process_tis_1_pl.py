import numpy as np
from tensorflow import keras
import tensorflow as tf
from pyspark.sql import SparkSession
from petastorm import TransformSpec
import os
from os.path import dirname, abspath
from loguru import logger
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pr3d.de import GammaEVM
import warnings
warnings.filterwarnings("ignore")

# Based on this https://databricks.com/blog/2020/06/16/simplify-data-conversion-from-apache-spark-to-tensorflow-and-pytorch.html

"""
The purpose of this script is to train a GammaEVM model over service_delays
"""

def init_spark():

    spark = SparkSession.builder \
       .appName("Training") \
       .config("spark.driver.memory", "112g") \
       .config("spark.driver.maxResultSize",0) \
       .getOrCreate()

    sc = spark.sparkContext
    return spark,sc

# init Spark
spark,sc = init_spark()

npdtype = np.float64
tfdtype = tf.float64
strdtype = 'float64'

results_path = 'results/'
raw_dfs_path = 'raw_dfs/'
all_files = os.listdir(results_path+raw_dfs_path)
files = []
for f in all_files:
    if f.endswith(".parquet"):
        files.append(results_path + raw_dfs_path + f)

logger.info(f"Found these files at {results_path+raw_dfs_path}: {files}")

# read all files into Spark df
df=spark.read.parquet(*files)

dataset_size = 1024*120 # total will be the_number_of_files * file_samples_size
batch_size = 1024
df = df.limit(dataset_size)
df_train, df_val = df.randomSplit([0.9, 0.1], seed=12345)
# Make sure the number of partitions is at least the number of workers which is required for distributed training.
NUM_WORKERS = 1
df_train = df_train.repartition(NUM_WORKERS)
df_val = df_val.repartition(NUM_WORKERS)

# Set a cache directory on DBFS FUSE for intermediate data.
file_path = dirname(abspath(__file__))
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, 'file://' + file_path + '/sparkcache')
logger.info(f"Spark cache folder is set up at: {'file://' + file_path + '/sparkcache'}")

converter_train = make_spark_converter(df_train)
converter_val = make_spark_converter(df_val)

logger.info(f"Parquet files loaded, train sampels: {len(converter_train)}, validation samples: {len(converter_val)}")

def transform_row(pd_batch):
    """
    The input and output of this function are pandas dataframes.
    """
    #print(pd_batch)
    pd_batch = pd_batch[['service_delay']]
    pd_batch['y_input'] = pd_batch['service_delay']
    pd_batch['dummy_input'] = 0.00
    pd_batch = pd_batch.drop(columns=['service_delay'])
    #print(new_batch)
    return pd_batch

# Note that the output shape of the `TransformSpec` is not automatically known by petastorm, 
# so we need to specify the shape for new columns in `edit_fields` and specify the order of 
# the output columns in `selected_fields`.
transform_spec_fn = TransformSpec(
  transform_row, 
  edit_fields=[('y_input', npdtype, (), False), ('dummy_input', npdtype, (), False)],
  selected_fields=['y_input', 'dummy_input']
)

# initiate the non conditional predictor
model = GammaEVM(
    #centers= 8,
    dtype = strdtype,
    bayesian = False,
    #batch_size = 1024,
)

with converter_train.make_tf_dataset(
    transform_spec=transform_spec_fn, 
    batch_size=batch_size,
) as train_dataset, \
    converter_val.make_tf_dataset(
    transform_spec=transform_spec_fn, 
    batch_size=batch_size,
) as val_dataset:

    # tf.keras only accept tuples, not namedtuples
    train_dataset = train_dataset.map(lambda x: ({'y_input':x.y_input, 'dummy_input':x.dummy_input},x.y_input))
    steps_per_epoch = len(converter_train) // batch_size

    val_dataset = val_dataset.map(lambda x: ({'y_input':x.y_input, 'dummy_input':x.dummy_input},x.y_input))
    validation_steps = max(1, len(converter_val) // batch_size)


    model._pl_training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01), 
        loss=model.loss,
    )

    logger.info(f"steps_per_epoch: {steps_per_epoch}, validation_steps: {validation_steps}")

    hist = model._pl_training_model.fit(
        train_dataset, 
        steps_per_epoch=steps_per_epoch,
        epochs=100,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        verbose=2
    )

model.save(results_path+"service_delay_model_pl.h5")

logger.info(f"Model saved, bayesian: {model.bayesian}, batch_size: {model.batch_size}")
logger.info(f"Parameters: {model.get_parameters()}")