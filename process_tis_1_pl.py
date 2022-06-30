import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
import tensorflow_io as tfio

from pr3d.de import GammaEVM
from dsio import load_parquet, parquet_tf_pipeline_unconditional_mutiple_file, parquet_tf_pipeline_unconditional_single_file2
import os


dtype = 'float32' # 'float32' or 'float16'

results_path = 'results/'
raw_dfs_path = 'raw_dfs/'
all_files = os.listdir(results_path+raw_dfs_path)
files = []
for f in all_files:
    if f.endswith(".parquet"):
        files.append(results_path + raw_dfs_path + f)

print(files)

dataset_size = 1024*90 # total will be the_number_of_files * file_samples_size
train_size = 1024*80
batch_size = 1024

# dataset pipeline
#train_dataset = parquet_tf_pipeline_unconditional_mutiple_file(
    #file_addrs = results_path + raw_dfs_path + "*.parquet",
#    file_addrs = files,
#    dummy_feature_name = "dummy_input",
#    label_name = "service_delay",
#    dataset_size = dataset_size,
#    train_size = train_size,
#    batch_size = batch_size,
#)

df = pd.read_parquet(files[0], engine='pyarrow')
print(len(df))

df_tfio = tfio.IODataset.from_parquet(filename = files[0])
ds = df_tfio.take(len(df))
print(df_tfio)
i = 0
for d in ds:
    i = i + 1
    print(i)

exit(0)

train_dataset = parquet_tf_pipeline_unconditional_single_file2(
    file_addr = files[0],
    dummy_feature_name = "dummy_input",
    label_name = "service_delay",
    dataset_size = dataset_size,
    train_size = train_size,
    batch_size = batch_size,
)

# initiate the non conditional predictor
model = GammaEVM(
    #centers= 8,
    dtype = dtype,
    bayesian = False,
    #batch_size = 1024,
)

model.fit_pipeline(
    train_dataset,
    test_dataset=None,
    optimizer = keras.optimizers.Adam(learning_rate=0.01),
    batch_size = 1024,
    epochs = 10, #1000
)

exit(0)

model.save(results_path+"service_delay_model.h5")

print(f"Model saved, bayesian: {model.bayesian}, batch_size: {model.batch_size}")
print(f"Parameters: {model.get_parameters()}")