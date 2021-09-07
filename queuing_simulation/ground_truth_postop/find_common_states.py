from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.window import Window
import pyspark.sql.functions as f
import sys

def init_spark():

    spark = SparkSession.builder \
       .appName("LoadParquets") \
       .config("spark.driver.memory", "8g") \
       .getOrCreate()

    sc = spark.sparkContext
    return spark,sc

def main():
    spark,sc = init_spark()

    #paths=['file1.parquet','file2.parquet']
    paths = ['../queuing_simulation/noaqm_threehop_parallel/saves/sim3hop_4_dataset_06_Sep_2021_15_14_26.parquet' \
            , '../queuing_simulation/noaqm_threehop_parallel/saves/sim3hop_2_dataset_06_Sep_2021_13_26_17.parquet'];

    df=spark.read.parquet(*paths)

    #df.show()
    #df.summary().show() #takes a very long time
    print(df.rdd.getNumPartitions())

    bu_df = df.select(['end2enddelay','h1_uplink_netstate','h1_compute_netstate','h1_downlink_netstate']).withColumnRenamed('end2enddelay', 'delay').withColumnRenamed('end2enddelay', 'delay').withColumnRenamed('h1_uplink_netstate', 'state1').withColumnRenamed('h1_compute_netstate', 'state2').withColumnRenamed('h1_downlink_netstate', 'state3');

    bc_df = df.select(['totaldelay_compute','totaldelay_downlink','h2_compute_netstate','h2_downlink_netstate']);
    bc_df = bc_df.withColumn('delay', bc_df['totaldelay_compute'] +  bc_df['totaldelay_downlink']).drop('totaldelay_compute').drop('totaldelay_downlink').withColumnRenamed('h2_compute_netstate', 'state1').withColumnRenamed('h2_downlink_netstate', 'state2');

    bd_df = df.select(['totaldelay_downlink','h3_downlink_netstate']).withColumnRenamed('totaldelay_downlink', 'delay').withColumnRenamed('h3_downlink_netstate', 'state1');

    #bu_df.show()
    #bc_df.show()
    #bd_df.show()

    common_bu_df = bu_df.groupBy('state1','state2','state3').count() \
                                                            .sort(col('count').desc()) \
                                                            .withColumn('prob', col('count')/bu_df.count());
    #                                                        .withColumn('cumprob', f.sum(col('prob')).over(Window.partitionBy().orderBy().rowsBetween(-sys.maxsize, 0)));

    common_bc_df = bu_df.groupBy('state1','state2').count() \
                                                   .sort(col('count').desc()) \
                                                   .withColumn('prob', col('count')/bu_df.count());
    #                                               .withColumn('cumprob', f.sum(col('prob')).over(Window.partitionBy().orderBy().rowsBetween(-sys.maxsize, 0)));

    common_bd_df = bu_df.groupBy('state1').count() \
                                          .sort(col('count').desc()) \
                                          .withColumn('prob', col('count')/bu_df.count());
    #                                     .withColumn('cumprob', f.sum(col('prob')).over(Window.partitionBy().orderBy().rowsBetween(-sys.maxsize, 0)));

    common_bu_df.limit(60).toPandas().to_csv('common_bu_df.csv', index=False);
    common_bc_df.limit(30).toPandas().to_csv('common_bc_df.csv', index=False);
    common_bd_df.limit(15).toPandas().to_csv('common_bd_df.csv', index=False);

if __name__ == '__main__':
  main()
