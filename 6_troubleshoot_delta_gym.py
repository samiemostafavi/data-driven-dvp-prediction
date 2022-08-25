import os
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.ticker as mticker
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import spark_partition_id
from pr3d.de import ConditionalGammaEVM, ConditionalGaussianMM, ConditionalGammaMixtureEVM
from pyspark.sql.types import StructType, StructField, FloatType
import tensorflow as tf
import warnings
from loguru import logger

warnings.filterwarnings("ignore")

def init_spark():

    # "spark.driver.memory" must not exceed the total memory of the device: SWAP + RAM
    # "spark.sql.execution.arrow.pyspark.enabled" is for faster conversion of Pandas dataframe to spark

    spark = SparkSession.builder \
        .master("local") \
        .appName("LoadParquets") \
        .config("spark.executor.memory","6g") \
        .config("spark.driver.memory", "70g") \
        .config("spark.driver.maxResultSize",0) \
        .getOrCreate()

    sc = spark.sparkContext
    return spark,sc

# init Spark
spark,sc = init_spark()


# open the ground truth empirical datasets
project_path = 'projects/qlen_benchmark/'
records_path = 'projects/qlen_benchmark/records'
condition_labels = ['queue_length']
key_label = 'end2end_delay'
conditions = {
    '0_results':0, 
    '1_results':1, 
    '2_results':2, 
    '3_results':3, 
    '4_results':4, 
    '5_results':5,
    '6_results':6, 
    '7_results':7, 
    '8_results':8, 
    '9_results':9, 
    '10_results':10, 
    '11_results':11,
    '12_results':12, 
    '13_results':13, 
    '14_results':14,
}


df_arr = {}
for results_dir in conditions:
    q_len = conditions[results_dir]
    cond_records_path = records_path + '/' + results_dir
    all_files = os.listdir(cond_records_path)
    files = []
    for f in all_files:
        if f.endswith(".parquet"):
            files.append(cond_records_path + '/'  + f)

    df=spark.read.parquet(*files)
    logger.info(f"Parquet files in {cond_records_path} are loaded.")
    logger.info(f"Total number of samples in this empirical dataset: {df.count()}")
    df_arr[str(q_len)] = df

# open the predictor
model_addr = project_path + '/predictors/' + 'gmevm_model_0.h5'
# load the non conditional predictor
pr_model = ConditionalGammaMixtureEVM( #ConditionalGaussianMM
    h5_addr = model_addr,
)
logger.info(f"Predictor: {Path(model_addr).stem} is loaded.")


# open JSON troubleshoot file
debug_json_addr = project_path + '/delta_debug_big.json'
with open(debug_json_addr,'r') as json_file:
    debug_dict = json.loads(json_file.read())

logger.info(F"Opened JSON file with {len(debug_dict)} drop records.")
total_drops = 0
for idx,item in enumerate(debug_dict):
    logger.info(f"Drop {idx}:")
    for dict_key in item.keys():
        logger.info(f"{dict_key}:")
        dict_df = pd.DataFrame(item[dict_key])
        # iterate over the rows
        emp_success_probs = []
        for row_idx, row in dict_df.iterrows():
            queue_length = row['queue_length']
            delay_budget = row['delay_budget']

            # get the conditional dataset
            cond_df = df_arr[str(int(queue_length))].alias('cond_df')
            total_count = cond_df.count()
            cond_df = cond_df \
                .where( cond_df['end2end_delay'] <= delay_budget )
            success_count = cond_df.count()
            emp_success_prob = success_count/total_count

            logger.info(f"Row {row_idx}, queue_length: {queue_length}, total samples {total_count}, successful samples {success_count}, empirical success_prob: {emp_success_prob:.3f}, predicted success_prob: {row['success_prob']:.3f}")

            emp_success_probs.append(emp_success_prob)
            #row['delay_budget']
        
        dict_df['emp_success_prob'] = emp_success_probs
        item[dict_key] = dict_df.to_dict()
    
    s1 = sum(list(item['orig']['emp_success_prob'].values()))
    s2 = sum(list(item['dropped']['emp_success_prob'].values()))
    logger.info(f"Dropping decision {idx}, s1: {s1}, s2: {s2}, delta: {s2-s1}, DROP: {s2>s1} ")
    if s2>s1:
        total_drops += 1

logger.info(f"Total drops: {total_drops}")

exit(0)

# draw prediction samples
# get pyspark dataframe in batches [https://stackoverflow.com/questions/60645256/how-do-you-get-batches-of-rows-from-spark-using-pyspark]
count = cond_df_nodelay.count() 
partitions = int(count/batch_size)
cond_df_nodelay = cond_df_nodelay.repartition(partitions)
# add the partition_number as a column
cond_df_nodelay = cond_df_nodelay.withColumn('partition_num', spark_partition_id())
cond_df_nodelay.persist()

total_partition = [int(row.partition_num) for row in 
    cond_df_nodelay.select('partition_num').distinct().collect()]

# create an empty pyspark dataset to append to it
pred_df = spark.createDataFrame([], schema = StructType([
    StructField(label, FloatType(), True) for label in [*condition_labels, key_label]
]))

i = 0
for each_df in total_partition:
    logger.info(f"Sample production progress: {100.0*i/len(total_partition):.2f}%, size of the resulting dataframe: {0 if i is 0 else pred_df.count()}")
    x_input_pd = cond_df_nodelay.where(cond_df_nodelay.partition_num == each_df) \
        .select(*condition_labels) \
        .toPandas()

    x_input_dict = { label:x_input_pd[label].to_numpy() for label in condition_labels }

    pred_np = pr_model.sample_n(
        x_input_dict,
        seed = seed,
    )
    doc = spark.createDataFrame(pd.DataFrame({**x_input_dict, key_label:pred_np}))
    pred_df = pred_df.union(doc)
    i = i+1

logger.info(F"Generated {pred_df.count()} prediction samples.")
#print(pred_df.count())
#print(pred_df.show())


# figure 1
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
ax = axes
minx = float('inf')
maxx = 0.0
minx = min(minx,*emp_quantile_values)
maxx = max(maxx,*emp_quantile_values)
# calculate the prediction quantile values
pred_quantile_values = list(pred_df.approxQuantile(key_label,[q[1] for q in quantile_labels],0))
minx = min(minx,*pred_quantile_values)
maxx = max(maxx,*pred_quantile_values)

# plot them!
ax.loglog(
    emp_quantile_values,
    1.00-np.array([t[1] for t in quantile_labels]),
    marker='.',
    label='simulation',
    #linestyle = 'None',
)
ax.loglog(
    pred_quantile_values,
    1.00-np.array([t[1] for t in quantile_labels]),
    marker='.',
    label='prediction',
)

# fix x axis and labels
#ax.set_xticks(range(math.ceil(minx),math.floor(maxx),100))
ax.set_xticks(np.logspace(math.log10(minx),math.log10(maxx),10 ))
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
ax.set_xlabel('Latency [log]')
ax.set_ylabel('Tail probability [log]')
ax.grid()
ax.legend()

# figure out the title 
sentence = [
    f"{label}={conditions[label]}" 
        for c,label in enumerate(condition_labels)
]
sentence = ','.join(sentence)
ax.set_title(sentence)


# figure 2
fig.tight_layout()
plt.savefig('validation_p30_0.png')

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
ax = axes

emp_cond_df_list = cond_df.select(key_label). \
      rdd.flatMap(lambda x: x).collect()

pred_cond_df_list = pred_df.select(key_label). \
      rdd.flatMap(lambda x: x).collect()

cond_df_list = [[x,y]for x,y in zip(emp_cond_df_list,pred_cond_df_list)]
cond_df_list = pd.DataFrame(cond_df_list,columns=['simulation','prediction'])
sns.histplot(
    cond_df_list['simulation'],
    kde=True,
    ax = ax,
    stat="density",
    label='simulation',
    color='b',
)

sns.histplot(
    cond_df_list['prediction'],
    kde=True,
    ax = ax,
    stat="density",
    label='prediction',
    color='r',
)

# figure out the title 
sentence = [
    f"{label}={conditions[label]}" 
        for c,label in enumerate(condition_labels)
]
sentence = ','.join(sentence)

ax.set_xlim(0,list(cond_df.approxQuantile(key_label,[0.9],0))[0])
ax.set_title(sentence)
ax.set_xlabel('Latency')
ax.set_ylabel('Probability')
ax.grid()
ax.legend()

fig.tight_layout()
plt.savefig('validation_p30_1.png')