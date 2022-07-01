import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

from pr3d.de import GammaEVM
from dsio import load_parquet
import os

dtype = 'float64' # 'float32' or 'float16'

results_path = 'results/'
raw_dfs_path = 'raw_dfs/'
all_files = os.listdir(results_path+raw_dfs_path)
files = []
for f in all_files:
    if f.endswith(".parquet"):
        files.append(results_path + raw_dfs_path + f)

files = [files[0],files[1],files[2],files[3]]
df = load_parquet(file_addresses=files,read_columns=['service_delay'])
print(df)
print(len(df))

# initiate the non conditional predictor
model = GammaEVM(
    #centers= 8,
    dtype = dtype,
    bayesian = False,
    #batch_size = 1024,
)

# shuffle the rows
training_df = df.sample(frac=1)

Y = training_df['service_delay'].to_numpy()

# train the model (it must be divisible by batch_size)
training_samples_num = 1024*100

# training
model.fit(
    Y[0:training_samples_num],
    batch_size = 1024, # 1024 training_samples_num
    epochs = 200, # 1000, 5000
    optimizer = keras.optimizers.Adam(learning_rate=0.01),
)


model.save(results_path+"service_delay_model.h5")

print(f"Model saved, bayesian: {model.bayesian}, batch_size: {model.batch_size}")
print(f"Parameters: {model.get_parameters()}")