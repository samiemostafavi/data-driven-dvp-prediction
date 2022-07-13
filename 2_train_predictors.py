import numpy as np
from tensorflow import keras
import tensorflow as tf
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

training_params = {
    'dataset_size': 60*1024, #60*1024*1024
    'batch_size': 1024, #1024*240
    'train_val_split': {
        'fraction': [0.99, 0.01],
        'seed': 12345,
    },
    'rounds' : [
        {'learning_rate': 0.01, 'epochs':10},
        {'learning_rate': 0.001, 'epochs':20},
    ],
}

condition_labels = ['queue_length', 'longer_delay_prob']
y_label = 'end2end_delay'

models_conf = [
    {
        'type': 'gmm',
        'bayesian': False,
        'ensembles': 1,
        'centers': 5,
        'hidden_sizes': (20, 50, 20),
        'condition_labels' : condition_labels,
        'y_label' : y_label,
        'training_params': training_params,
    },
    {
        'type':'gmevm',
        'bayesian':False,
        'ensembles':1,
        'centers':4,
        'hidden_sizes':(20, 50, 20),
        'condition_labels' : condition_labels,
        'y_label' : y_label,
        'training_params': training_params,
    },
]

# open the dataset
project_folder = "projects/ar_benchmark/"
project_paths = [project_folder+name for name in os.listdir(project_folder) if os.path.isdir(os.path.join(project_folder, name))]

#project_paths = [
#    'projects/tail_benchmark/p1_results',
#]

for project_path in project_paths:
    logger.info(f"Openning the path '{project_path}'")
    
    predictors_path = project_path + '/predictors/'
    os.makedirs(predictors_path, exist_ok=True)

    records_path = project_path + '/records/'
    all_files = os.listdir(records_path)
    files = []
    for f in all_files:
        if f.endswith(".parquet"):
            files.append(records_path + f)

    # read all files into Spark df
    main_df=spark.read.parquet(*files)


    for model_conf in models_conf:

        training_params = model_conf['training_params']

        # take the desired number of records for learning
        df = main_df.sample(
            withReplacement=False, 
            fraction=training_params['dataset_size']/main_df.count(),
        )
        df_train, df_val = df.randomSplit(
            training_params['train_val_split']['fraction'], 
            seed=training_params['train_val_split']['seed'],
        )

        # dataset partitioning and making the converters
        # Make sure the number of partitions is at least the number of workers which is required for distributed training.
        df_train = df_train.repartition(1)
        df_val = df_val.repartition(1)
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

        model_type = model_conf['type']
        ensembles = model_conf['ensembles']
        condition_labels = model_conf['condition_labels']
        training_rounds = training_params['rounds']
        batch_size = training_params['batch_size']
        for num_ensemble in range(ensembles):

            # initiate the non conditional predictor
            if model_type == 'gmm':
                model = ConditionalGaussianMM(
                    x_dim = condition_labels,
                    centers = model_conf['centers'],
                    hidden_sizes = model_conf['hidden_sizes'],
                    dtype = strdtype,
                    bayesian = model_conf['bayesian'],
                    #batch_size = 1024,
                )
            elif model_type == 'gevm':
                model = ConditionalGammaEVM(
                    x_dim = condition_labels,
                    hidden_sizes = model_conf['hidden_sizes'],
                    dtype = strdtype,
                    bayesian = model_conf['bayesian'],
                    #batch_size = 1024,
                )
            elif model_type == 'gmevm':
                model = ConditionalGammaMixtureEVM(
                    x_dim = condition_labels,
                    centers = model_conf['centers'],
                    hidden_sizes = model_conf['hidden_sizes'],
                    dtype = strdtype,
                    bayesian = model_conf['bayesian'],
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

                for idx, params in enumerate(training_rounds):
                    logger.info(f"Starting training session {idx}/{len(training_rounds)} with {params}")

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

            model.save(predictors_path+f"{model_type}_model_{num_ensemble}.h5")
            with open(predictors_path+f"{model_type}_model_{num_ensemble}.json", "w") as write_file:
                json.dump(model_conf, write_file, indent=4)

            logger.info(f"A {model_type} {'non-bayesian' if model.bayesian else 'bayesian'} " + \
                f"model got trained and saved, ensemble: {num_ensemble}.")