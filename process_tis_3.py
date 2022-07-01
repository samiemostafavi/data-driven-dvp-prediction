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
processed_dfs_path = 'processed_dfs/'
all_files = os.listdir(results_path+raw_dfs_path)

# load the fitted service delay model
service_delay_model = GammaEVM(
    h5_addr = results_path+"service_delay_model_pl.h5",
)
print("Service delay model is loaded. Parameters: {0}".format(service_delay_model.get_parameters()))

for f in all_files:
    if f.endswith(".parquet"):
        file_addr = results_path + raw_dfs_path + f
        df = load_parquet(
            file_addresses=[file_addr],
            read_columns=None,
        )
        # process the time_in_service and convert it to longer_delay_prob
        tis = np.squeeze(df[['time_in_service']].to_numpy())
        longer_delay_prob = np.float64(1.00)-service_delay_model.prob_batch(tis)[2]

        df['longer_delay_prob'] = longer_delay_prob
        # replace NaNs with zeros
        df['longer_delay_prob'] = df['longer_delay_prob'].fillna(np.float64(0.00))
        df.to_parquet(results_path + processed_dfs_path + f)

        print(f"File {f} loaded with size {len(tis)}, processed and saved.")