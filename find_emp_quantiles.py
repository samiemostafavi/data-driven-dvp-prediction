from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col,sequence
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType
from pyspark.sql.functions import monotonically_increasing_id
import numpy as np
import pandas as pd
import math
import os
import itertools
from tabulate import tabulate

results_path = 'results/'

def init_spark():

    spark = SparkSession.builder \
       .appName("LoadParquets") \
       .config("spark.driver.memory", "112g") \
       .config("spark.driver.maxResultSize",0) \
       .getOrCreate()

    sc = spark.sparkContext
    return spark,sc


def make_conditions_dict(states_conf):

    """
    states_conf = {
        'queue_length':{
            'n':2,
            'max':10,
        },
        'longer_delay_prob':{
            'n':2,
            'max':1,
        },
        'chert':{
            'n':2,
            'max':100,
        }
    }
    """

    for key in states_conf:
        n = states_conf[key]['n']
        max = states_conf[key]['max']
        quants = np.linspace(0, max, num=n+1)
        cond_values = [ (quant,quants[idx+1]) for idx, quant in enumerate(quants) if (idx+1)<len(quants) ]
        #print(f"{key} conditions: {cond_values}")
        states_conf[key]['cond_values'] = cond_values

    #print(states_conf)
    
    lists = []
    for key in states_conf:
        lists.append(states_conf[key]['cond_values'])

    conditions_table = pd.DataFrame(columns = [key for key in states_conf])
    p = list(itertools.product(*lists))
    for v in p:
        #print(list(v))
        conditions_table.loc[len(conditions_table)] = list(v)
        
    #print(tabulate(conditions_table, headers='keys', tablefmt='psql'))

    return states_conf,conditions_table


# init Spark
spark,sc = init_spark()

# set manually:
#qrange_list = [0.9, 0.99, 0.999, 0.9999, 0.99999];

# or automatic:
N_qt=10
qlim = [0.00001, 0.1]; #0.00001 (real tail), 0.99999 (close to zero)
qrange_list = [1-i for i in np.logspace(math.log10( qlim[1] ), math.log10( qlim[0] ) , num=N_qt)]

# print Quantile range list
print('Quantile range list:')
print(qrange_list)

# open the dataframe from parquet files
results_path = 'results/'
processed_dfs_path = 'processed_dfs/'
all_files = os.listdir(results_path+processed_dfs_path)
files = []
for f in all_files:
    if f.endswith(".parquet"):
        files.append(results_path + processed_dfs_path + f)

df=spark.read.parquet(*files)
df.show()
#df.summary().show() #takes a very long time
print(f"Number of imported samples: {df.count()}")


# Figure out the conditions
conditions_conf, conditions_table =  make_conditions_dict({
        'queue_length':{
            'n':5,
            'max':10,
        },
        'longer_delay_prob':{
            'n':5,
            'max':1,
        }
    }
)

print(f"Conditions table: \n{conditions_table}")

# figure out conditionals and quantiles
quantiles_table = pd.DataFrame(columns = [str(q) for q in qrange_list])
for index, row in conditions_table.iterrows():
    print(f"-------- Calculating condition {index+1}/{len(conditions_table)}: {row.values}")

    cond_df = df.alias('cond_df')
    for key in row.keys():
        cond_df = cond_df.where( df[key]>= row[key][0] ).where( df[key] < row[key][1] )

    print(f"- Pyspark found {cond_df.count()} conditional samples")
    res = cond_df.approxQuantile('end2end_delay',qrange_list,0)
    print(f"- Conditional quantiles: {res}")
    quantiles_table.loc[len(quantiles_table)] = list(res)


results = pd.concat([conditions_table, quantiles_table], axis=1)

print(tabulate(results, headers='keys', tablefmt='psql'))

results.to_csv(results_path + 'quantiles.csv')