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


df = load_parquet(file_addresses=files,read_columns=['service_delay'])

model = GammaEVM(
    h5_addr = results_path+"service_delay_model_pl.h5"
)

fig, ax = plt.subplots()

sns.histplot(
    df,
    kde=False, #True
    ax = ax,
    stat="density",
).set(title="count={0}".format(len(df)))
ax.title.set_size(10)
ax.set_xlim(0,40)

# then, plot predictions
y0, y1 = ax.get_xlim()  # extract the y endpoints
y_lin = np.linspace(y0, y1, 100, dtype=dtype)
if model.bayesian:
    for _ in range(100):
        pdf,log_pdf,ecdf = model.prob_batch(y = y_lin[:,None])
        ax.plot(y_lin, pdf, color='red', alpha=0.1, label='prediction')
else:
    pdf,log_pdf,ecdf = model.prob_batch(y = y_lin[:,None])
    ax.plot(y_lin, pdf, color='red', lw=2, label='prediction')

ax.legend(['Data', 'Prediction'])

fig.tight_layout()
plt.savefig(results_path+'service_delay_fit_pl.png')

