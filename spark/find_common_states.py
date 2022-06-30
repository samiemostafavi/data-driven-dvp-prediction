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
    paths = [ 'results/tailbench_0.parquet'  \
            , 'results/tailbench_1.parquet' ]

    df=spark.read.parquet(*paths)
    df = df.select(['end2end_delay','queue_length','service_delay','time_in_service'])
    
    df.show()
    #df.summary().show() #takes a very long time

    exit(0)

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