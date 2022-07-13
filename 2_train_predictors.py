import numpy as np
from tensorflow import keras
import tensorflow as tf
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from petastorm import TransformSpec
import os
from os.path import dirname, abspath
from loguru import logger
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pr3d.de import ConditionalGammaEVM, ConditionalGaussianMM, ConditionalGammaMixtureEVM
import warnings
warnings.filterwarnings("ignore")


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

condition_labels = ['queue_length', 'longer_delay_prob']
y_label = 'end2end_delay'

model_names = ['gmm','gmevm']
ensembles = 1

dataset_size = 60*1024*1024 # total will be the_number_of_files * file_samples_size
batch_size = 1024*240
train_val_split = {
    'fraction': [0.99, 0.01],
    'seed': 12345,
}
training_params = [
    {'learning_rate': 0.01, 'epochs':10},
    {'learning_rate': 0.001, 'epochs':20},
]

# open the dataframe from parquet files
project_folder = "projects/tail_benchmark/" 
project_paths = [project_folder+name for name in os.listdir(project_folder) if os.path.isdir(os.path.join(project_folder, name))]

# limit simulation folder
#project_paths = ['projects/tail_benchmark/p4_results']

for project_path in project_paths:
    logger.info(f"Starting simulation path: {project_path}")
    
    predictors_path = project_path + '/predictors/'
    os.makedirs(predictors_path, exist_ok=True)

    records_path = project_path + '/records/'
    all_files = os.listdir(records_path)
    files = []
    for f in all_files:
        if f.endswith(".parquet"):
            files.append(records_path + f)

    # read all files into Spark df
    df=spark.read.parquet(*files)

    # take desired number of records for learning
    df = df.sample(
        withReplacement=False, 
        fraction=dataset_size/df.count(),
    )
    df_train, df_val = df.randomSplit(train_val_split['fraction'], seed=train_val_split['seed'])

    # dataset partitioning and making the converters
    # Make sure the number of partitions is at least the number of workers which is required for distributed training.
    NUM_WORKERS = 1
    df_train = df_train.repartition(NUM_WORKERS)
    df_val = df_val.repartition(NUM_WORKERS)
    converter_train = make_spark_converter(df_train)
    converter_val = make_spark_converter(df_val)
    logger.info(f"Dataset loaded, train sampels: {len(converter_train)}, validation samples: {len(converter_val)}")

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

    for num_ensemble in range(ensembles):
        
        # limit
        #num_ensemble=2 

        for model_name in model_names:

            # initiate the non conditional predictor
            if model_name == 'gmm':
                model = ConditionalGaussianMM(
                    x_dim=condition_labels,
                    centers= 5,
                    hidden_sizes = (20, 50, 20),
                    dtype = strdtype,
                    bayesian = False,
                    #batch_size = 1024,
                )
            elif model_name == 'gevm':
                model = ConditionalGammaEVM(
                    x_dim=condition_labels,
                    centers=4,
                    hidden_sizes = (20, 50, 20),
                    dtype = strdtype,
                    bayesian = False,
                    #batch_size = 1024,
                )
            elif model_name == 'gmevm':
                model = ConditionalGammaMixtureEVM(
                    x_dim=condition_labels,
                    centers=4,
                    hidden_sizes = (20, 50, 50, 20),
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
                # map the dataset to the desired tf.keras input in _pl_training_model
                def map_fn(x):
                    x_dict = {}
                    for idx,cond in enumerate(condition_labels):
                        x_dict = {**x_dict, cond:x[idx]}
                    return ({**x_dict, 'y_input':x.y_input},x.y_input)

                train_dataset = train_dataset.map(map_fn)
                steps_per_epoch = len(converter_train) // batch_size

                val_dataset = val_dataset.map(map_fn)
                validation_steps = max(1, len(converter_val) // batch_size)

                for idx, params in enumerate(training_params):
                    logger.info(f"Starting training session {idx}/{len(training_params)} with {params}")

                    model._pl_training_model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']), 
                        loss=model.loss,
                    )

                    logger.info(f"steps_per_epoch: {steps_per_epoch}, validation_steps: {validation_steps}")

                    hist = model._pl_training_model.fit(
                        train_dataset, 
                        steps_per_epoch=steps_per_epoch,
                        epochs=params['epochs'],
                        validation_data=val_dataset,
                        validation_steps=validation_steps,
                        verbose=1,
                    )

                model.save(predictors_path+f"{model_name}_model_{num_ensemble}.h5")

                logger.info(f"Model {num_ensemble} saved, bayesian: {model.bayesian}, batch_size: {model.batch_size}")