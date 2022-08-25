import numpy as np
from tensorflow import keras
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from petastorm import TransformSpec
import os
from os.path import dirname, abspath
from loguru import logger
import json
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pr3d.de import ConditionalGammaEVM, ConditionalGaussianMM, ConditionalGammaMixtureEVM
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path


def init_spark():

    spark = SparkSession.builder \
       .appName("Training") \
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

npdtype = np.float64
tfdtype = tf.float64
strdtype = 'float64'


# open the ground truth written in the csv files
project_path = 'projects/tail_benchmark/p4_results'
condition_labels = ['queue_length', 'longer_delay_prob']
key_label = 'end2end_delay'
predictors_path = project_path + '/predictors/'
model_name_path = 'gmevm_model_0.h5'
model_addr = project_path + '/predictors/' + model_name_path
# load the non conditional predictor
model = ConditionalGammaMixtureEVM( #ConditionalGaussianMM #ConditionalGammaMixtureEVM
    h5_addr = model_addr,
)
seed = 12345
batch_size = 30000
logger.info(f"Predictor: {Path(model_addr).stem} is loaded.")

condition_labels = ['queue_length', 'longer_delay_prob']
y_label = 'end2end_delay'

training_params = {
    'dataset_size': 'all',#60*1024*1024,
    'batch_size': 1024*128,
    'rounds' : [
        #{'learning_rate': 0.01, 'epochs':20},
        {'learning_rate': 0.0001, 'epochs':10},
    ],
}

model_conf = {
    'type':'gmevm',
    'bayesian':False,
    'ensembles':1,
    'centers':4,
    'hidden_sizes':(20, 50, 20),
    'condition_labels' : condition_labels,
    'y_label' : y_label,
    'training_params': training_params,
}

logger.info(f"Openning the path '{project_path}'")


records_path = project_path + '/records/'
all_files = os.listdir(records_path)
files = []
for f in all_files:
    if f.endswith(".parquet"):
        files.append(records_path + f)

# read all files into Spark df
main_df=spark.read.parquet(*files)

# Absolutely necessary for randomizing the rows (bug fix)
# first shuffle, then sample!
main_df = main_df.orderBy(rand())

training_params = model_conf['training_params']

if training_params['dataset_size'] == 'all':
    df_train = main_df.sample(
        withReplacement=False, 
        fraction=1.00,
    )
else:
    # take the desired number of records for learning
    df_train = main_df.sample(
        withReplacement=False, 
        fraction=training_params['dataset_size']/main_df.count(),
    )

# dataset partitioning and making the converters
# Make sure the number of partitions is at least the number of workers which is required for distributed training.
df_train = df_train.repartition(1)
converter_train = make_spark_converter(df_train)
logger.info(f"Dataset loaded, train sampels: {len(converter_train)}")

def transform_row(pd_batch):
    """
    The input and output of this function are pandas dataframes.
    """
    
    pd_batch = pd_batch[[y_label,*condition_labels]]
    pd_batch['y_input'] = pd_batch[y_label]
    pd_batch = pd_batch.drop(columns=[y_label])
    return pd_batch

# Note that the output shape of the `TransformSpec` is not automatically known by petastorm, 
# so we need to specify the shape for new columns in `edit_fields` and specify the order of 
# the output columns in `selected_fields`.
x_fields = [ (cond, npdtype, (), False) for cond in condition_labels ]
transform_spec_fn = TransformSpec(
    transform_row,
    edit_fields = [
        *x_fields,
        ('y_input', npdtype, (), False),
    ],
    selected_fields=[*condition_labels,'y_input'],
)

model_type = model_conf['type']
ensembles = model_conf['ensembles']
condition_labels = model_conf['condition_labels']
training_rounds = training_params['rounds']
batch_size = training_params['batch_size']


with converter_train.make_tf_dataset(
    transform_spec=transform_spec_fn, 
    batch_size=batch_size,
) as train_dataset:

    # tf.keras only accept tuples, not namedtuples
    # map the dataset to the desired tf.keras input in _pl_training_model
    def map_fn(x):
        x_dict = {}
        for idx,cond in enumerate(condition_labels):
            x_dict = {**x_dict, cond:x[idx]}
        return ({**x_dict, 'y_input':x.y_input},x.y_input)

    train_dataset = train_dataset.map(map_fn)
    steps_per_epoch = len(converter_train) // batch_size

    for idx, params in enumerate(training_rounds):
        logger.info(f"Starting training session {idx}/{len(training_rounds)} with {params}")

        model._pl_training_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']), 
            loss=model.loss,
        )

        logger.info(f"steps_per_epoch: {steps_per_epoch}")

        hist = model._pl_training_model.fit(
            train_dataset, 
            steps_per_epoch=steps_per_epoch,
            epochs=params['epochs'],
            verbose=1,
        )

model.save(predictors_path+f"{model_type}_model_more.h5")
with open(predictors_path+f"{model_type}_model_more.json", "w") as write_file:
    json.dump(model_conf, write_file, indent=4)

logger.info(f"A {model_type} {'bayesian' if model.bayesian else 'non-bayesian'} " + \
    f"model got trained and saved.")