import os
import pandas as pd
from pathlib import Path
from functools import reduce
import polars as pl
from pr3d.de import ConditionalGammaEVM
from os.path import dirname, abspath
import tensorflow as tf
import warnings
from loguru import logger


# open the dataframe from parquet files
project_folder = "projects/tail_benchmark/" 
project_paths = [project_folder+name for name in os.listdir(project_folder) if os.path.isdir(os.path.join(project_folder, name))]

project_paths = ['projects/tail_benchmark/p3_results','projects/tail_benchmark/p4_results']

for project_path in project_paths:

    records_path = project_path + '/records_predicted/'
    all_files = os.listdir(records_path)
    files = []
    for f in all_files:
        if f.endswith(".parquet"):
            files.append(records_path + f)

    for idx,pred_records_addr in enumerate(files):
        logger.info(F"Working on {pred_records_addr}")
        df = pl.read_parquet(pred_records_addr)
        logger.info(f"Number of imported samples: {len(df)}")
        new_file_addr = records_path + Path(pred_records_addr).stem + "_n.parquet"
        logger.info(F"Saving {new_file_addr}")
        df.write_parquet(new_file_addr, compression = "snappy")
