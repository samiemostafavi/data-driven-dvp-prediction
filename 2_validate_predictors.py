import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.ticker as mticker
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import spark_partition_id
from pr3d.de import ConditionalGammaEVM
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


# open the ground truth written in the csv files
project_path = 'projects/tail_benchmark/p1_results'
condition_labels = ['queue_length', 'longer_delay_prob']
key_label = 'end2end_delay'
conditions = {
    'queue_length':(0.0,2.0),
    'longer_delay_prob' :(0.0,0.2),
}
cond_dataset_limit = 300000
quantiles_file_addr = project_path + '/quantiles.csv'
model_addr = project_path + '/predictors/' + 'model_0.h5'
# load the non conditional predictor
pr_model = ConditionalGammaEVM(
    h5_addr = model_addr,
)
rng = tf.random.Generator.from_seed(12345)
batch_size = 30000
logger.info(f"Predictor: {Path(model_addr).stem} is loaded.")


# open the empirical dataset
records_path = project_path + '/records/'
all_files = os.listdir(records_path)
files = []
for f in all_files:
    if f.endswith(".parquet"):
        files.append(records_path + f)

df=spark.read.parquet(*files)
logger.info(f"Project path: {project_path} is loaded.")
logger.info(f"Total number of samples in the empirical dataset: {df.count()}")

# get the conditional dataset
cond_df = df.alias('cond_df')
for cond_key in conditions.keys():
        cond_df = cond_df \
            .where( cond_df[cond_key]>= conditions[cond_key][0] ) \
            .where( cond_df[cond_key] < conditions[cond_key][1] )

cond_df = cond_df.sample(float(cond_dataset_limit/cond_df.count()),seed = 12345)
cond_df_nodelay = cond_df.select(*condition_labels)
logger.info(f"Applying conditions: {conditions} and length limit: {cond_dataset_limit} to the dataset.")
logger.info(f"Number of samples in the conditional empirical dataset: {cond_df.count()}")

# load empirical quantiles from CSV file
emp_quantiles_pd = pd.read_csv(quantiles_file_addr)
logger.info(f"Empirical quantiles loaded from the CSV file {quantiles_file_addr}.")
# find quantile labels
quantile_labels = []
for key in emp_quantiles_pd.keys():
    try:
        float(key)
    except:
        continue
    quantile_labels.append((key,np.float64(key)))
logger.info(f"Quantile points to benchmark the predictors: {[q[1] for q in quantile_labels]}")
# find condition index
for idx, key in enumerate(conditions.keys()):
    if idx == 0:
        rows = emp_quantiles_pd[ (emp_quantiles_pd[key] == str(conditions[key])) ]
    else:
        rows = rows[ (emp_quantiles_pd[key] == str(conditions[key])) ]

assert len(rows) == 1
condition_idx = rows.index[0]
logger.info(f"Condition index to refer to in the CSV file: {condition_idx}")

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
        rng = rng,
    )
    doc = spark.createDataFrame(pd.DataFrame({**x_input_dict, key_label:pred_np}))
    pred_df = pred_df.union(doc)
    i = i+1

print(pred_df.count())
print(pred_df.show())

# figure 1
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
ax = axes
minx = float('inf')
maxx = 0.0
quantiles_df = emp_quantiles_pd[[t[0] for t in quantile_labels]]
# read quantile values from the CSV dataframe
emp_quantile_values = quantiles_df.loc[condition_idx, :].values.tolist()
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
    f"{label}={emp_quantiles_pd.loc[condition_idx, label]}" 
        for c,label in enumerate(condition_labels)
]
sentence = ','.join(sentence)
ax.set_title(sentence)


# figure 2
fig.tight_layout()
plt.savefig('validation_0.png')

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
    f"{label}={emp_quantiles_pd.loc[condition_idx, label]}" 
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
plt.savefig('validation_1.png')