from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col,sequence
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType
from pyspark.sql.functions import monotonically_increasing_id
import numpy as np
import math

def init_spark():

    spark = SparkSession.builder \
       .appName("LoadParquets") \
       .config("spark.driver.memory", "112g") \
       .config("spark.driver.maxResultSize",0) \
       .getOrCreate()

    sc = spark.sparkContext
    return spark,sc

def main():
    spark,sc = init_spark()

    #qrange_list = [0.9, 0.99, 0.999, 0.9999, 0.99999];
    N_qt=10
    qlim = [0.00001, 0.01]; #0.00001 (real tail), 0.99999 (close to zero)
    qrange_list = [1-i for i in np.logspace(math.log10( qlim[1] ), math.log10( qlim[0] ) , num=N_qt)]
    print('Quantile range list:')
    print(qrange_list)

    #paths=['file1.parquet','file2.parquet']
    paths = [ '../queuing_simulation/noaqm_threehop_parallel/saves/sim3hop_4_dataset_06_Sep_2021_15_14_26.parquet'  \
            , '../queuing_simulation/noaqm_threehop_parallel/saves/sim3hop_6_dataset_06_Sep_2021_17_51_38.parquet'  \
            , '../queuing_simulation/noaqm_threehop_parallel/saves/sim3hop_2_dataset_06_Sep_2021_13_26_17.parquet'  \
            , '../queuing_simulation/noaqm_threehop_parallel/saves/sim3hop_10_dataset_06_Sep_2021_22_35_22.parquet' \
            , '../queuing_simulation/noaqm_threehop_parallel/saves/sim3hop_8_dataset_06_Sep_2021_20_32_44.parquet'];

    df=spark.read.parquet(*paths)

    common_bu_df = spark.read.option("header",True) \
                             .csv("common_bu_df.csv");
    common_bc_df = spark.read.option("header",True) \
                             .csv("common_bc_df.csv");
    common_bd_df = spark.read.option("header",True) \
                             .csv("common_bd_df.csv");

    #df.show()
    #df.summary().show() #takes a very long time
    print(df.count())
    print(df.rdd.getNumPartitions())

    bu_df = df.select(['end2enddelay','h1_uplink_netstate','h1_compute_netstate','h1_downlink_netstate']).withColumnRenamed('end2enddelay', 'delay').withColumnRenamed('end2enddelay', 'delay').withColumnRenamed('h1_uplink_netstate', 'state1').withColumnRenamed('h1_compute_netstate', 'state2').withColumnRenamed('h1_downlink_netstate', 'state3');

    bc_df = df.select(['totaldelay_compute','totaldelay_downlink','h2_compute_netstate','h2_downlink_netstate']);
    bc_df = bc_df.withColumn('delay', bc_df['totaldelay_compute'] +  bc_df['totaldelay_downlink']).drop('totaldelay_compute').drop('totaldelay_downlink').withColumnRenamed('h2_compute_netstate', 'state1').withColumnRenamed('h2_downlink_netstate', 'state2');

    bd_df = df.select(['totaldelay_downlink','h3_downlink_netstate']).withColumnRenamed('totaldelay_downlink', 'delay').withColumnRenamed('h3_downlink_netstate', 'state1');

    print('Seperate predictor dataframes created.')

    #bu_df.show()
    #bc_df.show()
    #bd_df.show()

    # filter rows by common states
    #cond_bu_df = bu_df.where((bu_df.state1=='1') & (bu_df.state2=='2') & (bu_df.state3=='2'));
    #cond_bu_df.summary().show();

    # show the quantiles
    #print(cond_bu_df.approxQuantile('delay',qrange_list,0));

 
    print('Before uplink:')
    rowList = [];
    for row in common_bu_df.rdd.collect():
        res = bu_df.where((bu_df.state1==row['state1']) & (bu_df.state2==row['state2']) & (bu_df.state3==row['state3'])).approxQuantile('delay',qrange_list,0)
        print(res)
        rowList.append(res)
    
    rdd1 = sc.parallelize(rowList)
    row_rdd = rdd1.map(lambda x: Row(x))
    sqlContext = SQLContext(sc)
    new_df=sqlContext.createDataFrame(row_rdd,['quants'])
    dlist = new_df.columns
    new_df = new_df.select(dlist+[(col("quants")[x]).alias(str(qrange_list[x])) for x in range(0, len(qrange_list))]).drop('quants')
    common_bu_df.toPandas().join(new_df.toPandas()).to_csv('common_bu_df_quants.csv', index=False);


    #schema = StructType(common_bu_df.schema.fields + new_df.schema.fields)
    #df1df2 = common_bu_df.rdd.zip(new_df.rdd).map(lambda x: x[0]+x[1])
    #new_df = spark.createDataFrame(df1df2, schema)


    print('Before compute:')
    rowList = [];
    for row in common_bc_df.rdd.collect():
        res = bc_df.where((bc_df.state1==row['state1']) & (bc_df.state2==row['state2'])).approxQuantile('delay',qrange_list,0)
        print(res)
        rowList.append(res)
    
    rdd1 = sc.parallelize(rowList)
    row_rdd = rdd1.map(lambda x: Row(x))
    sqlContext = SQLContext(sc)
    new_df=sqlContext.createDataFrame(row_rdd,['quants'])
    dlist = new_df.columns
    new_df = new_df.select(dlist+[(col("quants")[x]).alias(str(qrange_list[x])) for x in range(0, len(qrange_list))]).drop('quants')
    common_bc_df.toPandas().join(new_df.toPandas()).to_csv('common_bc_df_quants.csv', index=False);


    print('Before downlink:')
    rowList = [];
    for row in common_bd_df.rdd.collect():
        res = bd_df.where((bd_df.state1==row['state1'])).approxQuantile('delay',qrange_list,0);
        print(res)
        rowList.append(res)

    rdd1 = sc.parallelize(rowList)
    row_rdd = rdd1.map(lambda x: Row(x))
    sqlContext = SQLContext(sc)
    new_df=sqlContext.createDataFrame(row_rdd,['quants'])
    dlist = new_df.columns
    new_df = new_df.select(dlist+[(col("quants")[x]).alias(str(qrange_list[x])) for x in range(0, len(qrange_list))]).drop('quants')
    common_bd_df.toPandas().join(new_df.toPandas()).to_csv('common_bd_df_quants.csv', index=False);

    #print(rowList);

if __name__ == '__main__':
  main()
