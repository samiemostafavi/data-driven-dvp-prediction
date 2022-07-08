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
from pyspark.sql import SparkSession
from pyspark.sql.functions import spark_partition_id
from pyspark.sql import Window
from pr3d.de import ConditionalGammaEVM
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.sql.types import StructType,StructField, FloatType
from petastorm import TransformSpec
from os.path import dirname, abspath
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

def init_spark():

    # "spark.driver.memory" must not exceed the total memory of the device: SWAP + RAM

    spark = SparkSession.builder \
       .appName("LoadParquets") \
       .config("spark.executor.memory","6g") \
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
print(f"Spark cache folder is set up at: {'file://' + file_path + '/sparkcache'}")

# open the ground truth written in the csv files
project_folder = "projects/tail_benchmark/"
project_paths = [project_folder+name for name in os.listdir(project_folder) if os.path.isdir(os.path.join(project_folder, name))]
project_paths = ['projects/tail_benchmark/p2_results','projects/tail_benchmark/p1_results']

condition_labels = ['queue_length', 'longer_delay_prob']

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
        'model' : model,
        'dataframe' : dataframes[idx],
        'project_path' : project_paths[idx],
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
            sample_key = key
            #print(simulation_results[key]['dataframe'].keys())

# fix axis and titles
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
        sample_df = simulation_results[sample_key]['dataframe']
        sentence = [
            f"{label}={sample_df.loc[idx, label]}" 
                for c,label in enumerate(condition_labels)
        ]
        sentence = ','.join(sentence)
        ax.set_title(sentence)


# Predictors benchmarking
# open the dataframe from parquet files
for project_path in project_paths:
    records_path = project_path + '/records/'
    all_files = os.listdir(records_path)
    files = []
    for f in all_files:
        if f.endswith(".parquet"):
            files.append(records_path + f)

    df=spark.read.parquet(*files)

for i in range(n):
    for j in range(m):
        ax = axes[i,j]
        idx = i*n+j

        # convert condition string e.g. (2.0, 4.0) to tuple[float]
        sample_df = simulation_results[sample_key]['dataframe']
        conditions = {
            label : [ 
                float(element) 
                    for element in re.findall(
                        r'\((.*?),(.*?)\)',
                        sample_df.loc[idx, label]
                    )[0] 
            ]
                for c,label in enumerate(condition_labels)
        }


        for sim_key in simulation_results.keys():
            records_path = simulation_results[sim_key]['project_path'] + '/records/'
            all_files = os.listdir(records_path)
            files = []
            for f in all_files:
                if f.endswith(".parquet"):
                    files.append(records_path + f)

            df=spark.read.parquet(*files)

            # get the conditional dataset
            cond_df = df.alias('cond_df')
            for cond_key in conditions.keys():
                    cond_df = cond_df \
                        .where( cond_df[cond_key]>= conditions[cond_key][0] ) \
                        .where( cond_df[cond_key] < conditions[cond_key][1] )

            cond_df = cond_df.select(*condition_labels)

            print(f"The number of samples in the dataset: {cond_df.count()}")

            #print(f"{conditions} - {sim_key}")

            # open predictors
            records_path = simulation_results[sim_key]['project_path'] + '/predictors/'
            all_files = os.listdir(records_path)
            files = []
            for f in all_files:
                if f.endswith(".h5"):
                    files.append(records_path + f)

            # draw predictions
            # load the non conditional predictor
            pr_model = ConditionalGammaEVM(
                h5_addr = files[0],
            )
            rng = tf.random.Generator.from_seed(12345)
            batch_size = 1000000


            # get pyspark dataframe in batches [https://stackoverflow.com/questions/60645256/how-do-you-get-batches-of-rows-from-spark-using-pyspark]
            count = cond_df.count() 
            partitions = int(count/batch_size)
            cond_df = cond_df.repartition(partitions)
            # add the partition_number as a column
            cond_df = cond_df.withColumn('partition_num', spark_partition_id())
            cond_df.persist()

            total_partition = [int(row.partition_num) for row in 
                cond_df.select('partition_num').distinct().collect()]


            # create an empty pyspark dataset and append to it
            pred_cond_df = spark.createDataFrame([], schema = StructType([
                StructField('end2end_delay', FloatType(), True),
            ]))
            i = 0
            for each_df in total_partition:
                i = i+1
                cond_dataset_pd = cond_df.where(cond_df.partition_num == each_df) \
                    .select(*condition_labels) \
                    .toPandas()

                cond_dataset_dict = { label:cond_dataset_pd[label].to_numpy() for label in condition_labels }

                pred_cond_np = pr_model.sample_n(
                    cond_dataset_dict,
                    rng = rng,
                )
                print(i)
                doc = spark.createDataFrame(pd.DataFrame(pred_cond_np,columns=["end2end_delay"]))
                pred_cond_df = pred_cond_df.union(doc)
                print(pred_cond_df.count())

            print(pred_cond_df.count())
            print(pred_cond_df.show())

            exit(0)

            # convert the data from Spark to Tensorflow
            converter_cond_data = make_spark_converter(cond_df)
            transform_spec_fn = TransformSpec(
                func=None,
                edit_fields = [ 
                    (cond, np.float64, (), False) for cond in condition_labels 
                ],
                selected_fields=condition_labels,
            )
            

            

            with converter_cond_data.make_tf_dataset(
                transform_spec=transform_spec_fn, 
                batch_size=batch_size,
            ) as cond_dataset:

                # tf.keras only accept tuples, not namedtuples
                # map the dataset to the desired tf.keras input for params_model
                def map_fn(x):
                    res = {}
                    for idx,cond in enumerate(condition_labels):
                        res = {**res, cond:x[idx]}
                    return res

                cond_dataset = cond_dataset.map(map_fn)

                pred_cond_np = pr_model.sample_n(
                    cond_dataset,
                    batch_size = batch_size,
                    rng = rng,
                )
                doc = spark.createDataFrame(pred_cond_np, ["end2end_delay"])
                pred_cond_df = pred_cond_df.union(doc)
                print(pred_cond_df.show())
                exit(0)

                """
                u = 0
                #for cond_batch in cond_dataset:
                for i in range(10):
                    #print(cond_batch['longer_delay_prob'].numpy()) 
                    print(list(cond_dataset.take(10).as_numpy_iterator()))
                    u = u + 1   
                    print(u)
                    
                    pred_cond_np = pr_model.sample_n(
                        cond_batch,
                        batch_size = batch_size,
                        rng = rng,
                    )
                    # convert back the result to Pandas and then to Spark
                    pred_cond_pd = pd.DataFrame(pred_cond_np, columns=["end2end_delay"])
                    print(len(pred_cond_pd))
                    doc = spark.createDataFrame(pred_cond_pd, ["end2end_delay"])
                    pred_cond_df = pred_cond_df.union(doc)
                    print(pred_cond_df.count())
                """
            
            print(pred_cond_df)
            exit(0)

            x = [ np.squeeze(np.array(cond_df.select(label).collect())) for label in condition_labels  ]
            x = tuple(x)
            print(x)
            cond_pred_df = pr_model.sample_n(
                x = x,
                seed = 12345,
            )
            print(cond_pred_df)
            exit(0)
    

exit(0)

fig.tight_layout()
plt.savefig('smile.png')

