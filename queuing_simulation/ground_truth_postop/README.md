This folder contains the python scripts to collect the resulted parquet files and post process them:
1. find_common_states.py

Finds the most common states and writes them into .csv text files

2. find_quantiles.py

Reads the previous .csv file and searches through all the parquet files for the quantiles mentioned in the script.
If the total number of samples is large, modify the Spark memory config otherwise you will get "ran out of memory" OOM errors.

- Install PySpark and Numpy prior to running the code.

It is tested with python 3.6.9 and PySpark 3.1.2