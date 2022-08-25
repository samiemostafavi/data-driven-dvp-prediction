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
from loguru import logger

# open the ground truth written in the csv files
project_folder = "projects/tail_benchmark/"
project_paths = [project_folder+name for name in os.listdir(project_folder) if os.path.isdir(os.path.join(project_folder, name))]

condition_labels = ['queue_length', 'longer_delay_prob']

emp_quantiles_pds = []
pred_quantiles_pds = []
models = []
for project_path in project_paths:
    records_path = project_path + '/records/'

    # read the empirical quantiles file
    quantiles_file_addr = None
    for f in os.listdir(project_path):
        if f.endswith(".csv"):
            quantiles_file_addr = project_path + '/' + f
    assert quantiles_file_addr is not None
    emp_quantiles_pd = pd.read_csv(quantiles_file_addr)
    emp_quantiles_pds.append(emp_quantiles_pd)

    # read the predicted quantiles file
    # there can be multiple files for each results folder
    pds_tmp = []
    quantiles_file_addr = None
    for f in os.listdir(project_path + '/records_predicted/'):
        if f.endswith(".csv"):
            quantiles_file_addr = project_path + '/records_predicted/' + f
            with open(project_path + '/predictors/' + Path(quantiles_file_addr).stem + '.json') as json_file:
                pds_tmp.append({
                    'name' : Path(quantiles_file_addr).stem,
                    'model_json' : json.load(json_file),
                    'prediction_csv' : pd.read_csv(quantiles_file_addr),
                })

    assert quantiles_file_addr is not None
    pred_quantiles_pds.append(pds_tmp)
    
    # read the experiment model file to figure out what is different
    model_json = None
    for f in os.listdir(records_path):
        if f.endswith(".json"):
            with open(records_path+f) as json_file:
                model_json = json.load(json_file)
                break
    assert model_json is not None
    models.append(model_json)

logger.info(f"{len(emp_quantiles_pds)} simulation results found in {project_folder}.")
pretty = []
for idx,item in enumerate(pred_quantiles_pds):
    for pdict in item:
        pretty.append((idx, pdict['name']))

logger.info(f"Here are the predictors discovered for each of the simulations: {pretty}")

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
logger.info(f"The simulation parameter keys that I have found is: {keys_list}")


simulation_results = {}
for idx,model in enumerate(models):
    param_value = reduce(operator.getitem, keys_list, model)
    simulation_results[str(param_value)] = {
        'value_name' : keys_list[-1],
        'value' : param_value,
        'project_path' : project_paths[idx],
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

# limit
quantile_labels = quantile_labels[:-2]
logger.info(f"Quantile labels are {quantile_labels}")

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

logger.info(f"Benchmark conditions: {conditions}, total number of conditions: {len(conditions)}")

# plot the quantiles
n = 5
m = 5
assert n*m == len(conditions)
for key in simulation_results.keys():
    fig1, axes1 = plt.subplots(nrows=n, ncols=m, figsize=(m*7,n*5))
    fig2, axes2 = plt.subplots(nrows=n, ncols=m, figsize=(m*7,n*5))
    for i in range(n):
        for j in range(m):
            ax1 = axes1[i,j]
            ax2 = axes2[i,j]
            idx = i*n+j
        
            emp_quantiles_pd = simulation_results[key]['emp_quantile_pd'][[t[0] for t in quantile_labels]]
            emp_quantile_values = emp_quantiles_pd.loc[idx, :].values.tolist()

            emp_pd_dict = simulation_results[key]['emp_quantile_pd']
            num_samples_str = "{:.2e}".format(emp_pd_dict['num_samples'][idx])
            ax2.plot(
                1.00-np.array([t[1] for t in quantile_labels]),
                np.array(emp_quantile_values),
                marker='.', 
                label= f"simulation,{simulation_results[key]['value_name']}={simulation_results[key]['value']}, samples={num_samples_str}",
                #linestyle = 'None',
            )

            for pd_dict in simulation_results[key]['pred_quantile_pd']:

                pred_quantiles_pd = pd_dict['prediction_csv'][[t[0] for t in quantile_labels]]
                pred_quantile_values = pred_quantiles_pd.loc[idx, :].values.tolist()

                ax1.plot(
                    1.00-np.array([t[1] for t in quantile_labels]),
                    np.absolute(np.array(emp_quantile_values)-np.array(pred_quantile_values)),
                    marker='.', 
                    label= f"{pd_dict['name']},{simulation_results[key]['value_name']}={simulation_results[key]['value']}",
                    #linestyle = 'None',
                )

                num_samples_str = "{:.2e}".format(pd_dict['prediction_csv']['num_samples'][idx])
                ax2.plot(
                    1.00-np.array([t[1] for t in quantile_labels]),
                    np.array(pred_quantile_values),
                    marker='.', 
                    label= f"{pd_dict['name']}, {simulation_results[key]['value_name']}={simulation_results[key]['value']}, samples={num_samples_str}",
                    #linestyle = 'None',
                )
            
            # fix x axis
            #ax.set_xticks(range(math.ceil(minx),math.floor(maxx),100))
            ax1.set_xscale('log')
            ax1.set_xlabel('Quantiles [log]')
            ax1.set_ylabel('Error')

            # draw the legend
            ax1.legend()
            ax1.grid()

            # fix x axis
            #ax.set_xticks(range(math.ceil(minx),math.floor(maxx),100))
            ax2.set_xscale('log')
            ax2.set_xlabel('Quantiles [log]')
            ax2.set_ylabel('Error')

            # draw the legend
            ax2.legend()
            ax2.grid()

            # figure out the title 
            sentence = [
                f"{label}={sample_emp_quant_pd.loc[idx, label]}" 
                    for c,label in enumerate(condition_labels)
            ]

            sentence = ','.join(sentence)
            ax1.set_title(sentence)
            ax2.set_title(sentence)

            #print(simulation_results[key]['dataframe'].keys())
        

    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(simulation_results[key]['project_path'] + '/tail_err.png')
    fig2.savefig(simulation_results[key]['project_path'] + '/tail_values.png')

