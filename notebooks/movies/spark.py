# spark.py

import os
import csv 
from functools import wraps
from time import time
from typing import Tuple
import pandas as pd
from query import timeit, load_file, print_head
import pyspark


if __name__ == "__main__":
    files = ["links.csv", "ratings.csv","movies_metadata.csv"]

    # load files
    links = load_file(files[0])
    ratings = load_file(files[1])
    metas = load_file(files[2])


# load to spark
# test SparkSQL
# test Spark Dataframe join (distribute ?)

