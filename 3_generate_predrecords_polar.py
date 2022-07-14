import os
import pandas as pd
from pathlib import Path
from functools import reduce
import polars as pl

# On the servers that we work with, using CPU is much faster
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from pr3d.de import ConditionalGammaEVM, ConditionalGammaMixtureEVM, ConditionalGaussianMM
from os.path import dirname, abspath
import numpy as np
import json
import warnings
from loguru import logger
import glob


# open the ground truth written in the csv files
project_folder = "projects/tail_benchmark/"
project_paths = [project_folder+name for name in os.listdir(project_folder) if os.path.isdir(os.path.join(project_folder, name))]

# limit the simulations
#project_paths = ['projects/tail_benchmark/p1_results','projects/tail_benchmark/p3_results','projects/tail_benchmark/p4_results','projects/tail_benchmark/pz_results']
project_paths = ['projects/tail_benchmark/p1_results']

condition_labels = ['queue_length', 'longer_delay_prob']
key_label = 'end2end_delay'

# Predictors benchmarking
# open the dataframe from parquet files
for project_path in project_paths:
    records_path = project_path + '/records/'

    # open predictors
    predictors_path = project_path + '/predictors/'
    all_files = os.listdir(predictors_path)
    files = []
    for f in all_files:
        if f.endswith(".h5"):
            files.append(predictors_path + f)

    # limit the model
    files = [files[0]]

    records_pred_path= project_path + '/records_predicted/'
    os.makedirs(records_pred_path, exist_ok=True)
    for idx,model_addr in enumerate(files):
        logger.info(f"Working on {Path(model_addr).stem} of simulation {project_path}")
        with open(predictors_path + Path(model_addr).stem + '.json') as json_file:
            pr_model_json = json.load(json_file)

        # draw predictions
        # load the non conditional predictor
        model_type = pr_model_json['type']
        if model_type == 'gmm':
            pr_model = ConditionalGaussianMM(
                h5_addr = model_addr,
            )
            rng = np.random.default_rng(seed=12345)
        elif model_type == 'gevm':
            pr_model = ConditionalGammaEVM(
                h5_addr = model_addr,
            )
            rng = tf.random.Generator.from_seed(12345)
        elif model_type == 'gmevm':
            pr_model = ConditionalGammaMixtureEVM(
                h5_addr = model_addr,
            )
            rng = np.random.default_rng(seed=12345)
            
        batch_size = 100000

        df=pl.read_parquet(records_path+"*.parquet")
        df = df.select(condition_labels)
        logger.info(f"Loading records files with {len(df)} samples and batch size {batch_size}.")
        count = len(df)

        # create an empty pyspark dataset to append to it
        pred_df = None
        offset = 0
        while offset < count:
            logger.info(f"Sample production progress: {100.0*offset/count:.2f}%, size of the resulting dataframe: {0 if pred_df is None else len(pred_df)}")

            # take the right slice
            x_input_pd = df.slice(offset,batch_size).to_pandas()
            x_input_dict = { label:x_input_pd[label].to_numpy() for label in condition_labels }

            pred_np = pr_model.sample_n(
                x_input_dict,
                rng = rng,
            )

            doc = pl.DataFrame({**x_input_dict, key_label:pred_np})

            if offset == 0:
                pred_df = doc
            else:
                pred_df.vstack(doc,in_place=True)

            # update offset
            offset = offset + batch_size

        # very important to have compression = "snappy". Pyspark is not able to read l4z default compression of Polars
        pred_df.write_parquet(records_pred_path + Path(model_addr).stem + ".parquet", compression = "snappy")