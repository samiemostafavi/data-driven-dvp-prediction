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
import re

# open the ground truth written in the csv files
project_folder = "projects/tail_benchmark/"
project_paths = [project_folder+name for name in os.listdir(project_folder) if os.path.isdir(os.path.join(project_folder, name))]

condition_labels = ['queue_length', 'longer_delay_prob']

emp_quantiles_pds = []
pred_quantiles_pds = []
models = []
for project_path in project_paths:
    records_path = project_path + '/records/'

    # read the quantiles file
    quantiles_file_addr = None
    for f in os.listdir(project_path):
        if f.endswith(".csv"):
            quantiles_file_addr = project_path + '/' + f
    assert quantiles_file_addr is not None
    emp_quantiles_pd = pd.read_csv(quantiles_file_addr)
    emp_quantiles_pds.append(emp_quantiles_pd)

    # read the quantiles file
    quantiles_file_addr = None
    for f in os.listdir(project_path + '/records_predicted/'):
        if f.endswith(".csv"):
            quantiles_file_addr = project_path + '/records_predicted/' + f
    assert quantiles_file_addr is not None
    pred_quantiles_pd = pd.read_csv(quantiles_file_addr)
    pred_quantiles_pds.append(pred_quantiles_pd)

    # read the model file to figure out what is different
    model_json = None
    for f in os.listdir(records_path):
        if f.endswith(".json"):
            with open(records_path+f) as json_file:
                model_json = json.load(json_file)
                break
    assert model_json is not None
    models.append(model_json)

print(f"{len(emp_quantiles_pds)} simulation and prediction results found in {project_folder}.")
#print(dataframes.keys())
#print(models)

# find the simulation parameter
result = DeepDiff(
    t1 = models[0],
    t2 = models[1],
    exclude_paths={
        "root['name']",
    },
    exclude_regex_paths={r"root\['\w+'\].+\['seed'\]"}
)
result = result['values_changed']
keys_list = list(result.keys())[0] \
    .replace("']"," ") \
    .replace("['"," ") \
    .replace("["," ") \
    .replace("]"," ").split()
keys_list = list(map(lambda x:int(x) if x.isdigit() else x,keys_list))
keys_list = keys_list[1:]
print(f"The simulation parameter keys that I have found is: {keys_list}")

simulation_results = {}
for idx,model in enumerate(models):
    param_value = reduce(operator.getitem, keys_list, model)
    simulation_results[str(param_value)] = {
        'value_name' : keys_list[-1],
        'value' : param_value,
        'emp_quantile_pd' : emp_quantiles_pds[idx],
        'pred_quantile_pd' : pred_quantiles_pds[idx],
    }
simulation_results = dict(
    sorted(
        simulation_results.items(), 
        key=lambda item: float(item[0]),
    )
)
sample_sim_key = list(simulation_results.keys())[0]

# find quantile labels
quantile_labels = []
sample = list(simulation_results.values())[0]
for key in sample['emp_quantile_pd'].keys():
    try:
        float(key)
    except:
        continue
    quantile_labels.append((key,np.float64(key)))

# find conditions and their indexes in emp_quantiles_pd
# convert condition string e.g. (2.0, 4.0) to tuple[float]
sample_emp_quant_pd = simulation_results[sample_sim_key]['emp_quantile_pd']
conditions = []
for idx in range(len(sample_emp_quant_pd)):
    conditions.append({
        label : [ 
            float(element) for element in re.findall(
                r'\((.*?),(.*?)\)',
                sample_emp_quant_pd.loc[idx, label]
            )[0] 
        ] for c,label in enumerate(condition_labels)
    })

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
            emp_quantiles_pd = simulation_results[key]['emp_quantile_pd'][[t[0] for t in quantile_labels]]
            emp_quantile_values = emp_quantiles_pd.loc[idx, :].values.tolist()

            pred_quantiles_pd = simulation_results[key]['pred_quantile_pd'][[t[0] for t in quantile_labels]]
            pred_quantile_values = pred_quantiles_pd.loc[idx, :].values.tolist()

            ax.plot(
                1.00-np.array([t[1] for t in quantile_labels]),
                np.absolute(np.array(emp_quantile_values)-np.array(pred_quantile_values)),
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
        ax.set_xscale('log')
        ax.set_xlabel('Quantiles [log]')
        ax.set_ylabel('Error')

        # draw the legend
        ax.legend()
        ax.grid()

        # figure out the title 
        sentence = [
            f"{label}={sample_emp_quant_pd.loc[idx, label]}" 
                for c,label in enumerate(condition_labels)
        ]
        sentence = ','.join(sentence)
        ax.set_title(sentence)

fig.tight_layout()
plt.savefig('smile2.png')

