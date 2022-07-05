import os
import pandas as pd
from pathlib import Path
import json
from deepdiff import DeepDiff
from functools import reduce
import operator
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.ticker as mticker

# open the ground truth written in the csv files
project_folder = "projects/tail_benchmark/" 
project_paths = [project_folder+name for name in os.listdir(project_folder) if os.path.isdir(os.path.join(project_folder, name))]

condition_labels = ['queue_length', 'longer_delay_prob']
xlim = [1,1500]

dataframes = []
models = []

for project_path in project_paths:
    records_path = project_path + '/records/'

    # read the quantiles file
    quantiles_file_addr = None
    for f in os.listdir(project_path):
        if f.endswith(".csv"):
            quantiles_file_addr = project_path + '/' + f

    assert quantiles_file_addr is not None
    df = pd.read_csv(quantiles_file_addr)
    dataframes.append(df)

    # read the model file to figure out what is different
    model_json = None
    for f in os.listdir(records_path):
        if f.endswith(".json"):
            with open(records_path+f) as json_file:
                model_json = json.load(json_file)
                break
    assert model_json is not None
    models.append(model_json)

print(f"{len(dataframes)} simulation results found in {project_folder}.")
#print(dataframes.keys())
#print(models)

# find the simulation parameter
result = DeepDiff(models[0],models[1])
result = result['values_changed']
for key in result.keys():
    if ('seed' in key) or (key == "root['name']"):
        continue
    else:
        keys_list = key.replace("']"," ").replace("['"," ").split()
        break
keys_list = keys_list[1:]
print(f"The simulation parameter keys that I have found is: {keys_list}")

simulation_results = {}
for idx,model in enumerate(models):
    param_value = reduce(operator.getitem, keys_list, model)
    simulation_results[str(param_value)] = {
        'value_name' : keys_list[-1],
        'value' : param_value,
        'model' : model,
        'dataframe' : dataframes[idx],
    }

simulation_results = dict(
    sorted(
        simulation_results.items(), 
        key=lambda item: float(item[0]),
    )
)

# find quantile labels
quantile_labels = []
sample = list(simulation_results.values())[0]
for key in sample['dataframe'].keys():
    try:
        float(key)
    except:
        continue
    quantile_labels.append((key,np.float64(key)))


# plot the quantiles
n = 5
m = 5
fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(m*7,n*5))
minx = float('inf')
maxx = 0
for i in range(n):
    for j in range(m):
        ax = axes[i,j]
        idx = i*n+j
        for key in simulation_results.keys():
            quantiles_df = simulation_results[key]['dataframe'][[t[0] for t in quantile_labels]]
            quantile_values = quantiles_df.loc[idx, :].values.tolist()
            minx = min(minx,*quantile_values)
            maxx = max(maxx,*quantile_values)
            ax.loglog(
                quantile_values,
                1.00-np.array([t[1] for t in quantile_labels]),
                marker='.', 
                label= f"{simulation_results[key]['value_name']}={simulation_results[key]['value']}",
                #linestyle = 'None',
            )
            #print(simulation_results[key]['dataframe'].keys())

for i in range(n):
    for j in range(m):
        ax = axes[i,j]
        idx = i*n+j

        # fix x axis
        #ax.set_xticks(range(math.ceil(minx),math.floor(maxx),100))
        ax.set_xticks(np.logspace(math.log10(minx),math.log10(maxx),10 ))
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel('Latency [log]')
        ax.set_ylabel('Tail probability [log]')

        # draw the legend
        ax.legend()
        ax.grid()

        # figure out the title 
        sample_df = simulation_results[key]['dataframe']
        sentence = [
            f"{label}={sample_df.loc[idx, label]}" 
                for c,label in enumerate(condition_labels)
        ]
        sentence = ','.join(sentence)
        ax.set_title(sentence)

fig.tight_layout()
plt.savefig('smile.png')

