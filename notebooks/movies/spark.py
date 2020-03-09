# spark.py

import os
import csv 
from functools import wraps
from time import time
from typing import Tuple
import pandas as pd
from query import timeit, load_file, print_head
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as func

def mod(x):
    return (x, x%2)

@timeit
def load_files(links,ratings, metas):
    pdf_links = pd.read_csv(links)
    pdf_ratings = pd.read_csv(ratings)
    pdf_metas = pd.read_csv(metas).astype(str)
    df_links = spark.createDataFrame(pdf_links[['movieId','imdbId']])
    df_ratings = spark.createDataFrame(pdf_ratings[['userId','movieId','rating']])
    df_metas = spark.createDataFrame(pdf_metas[['imdb_id','title']])
    return df_links, df_ratings, df_metas

@timeit
def join(t1, t2, t3, c1, c2, c3, c4):
    t3 = t1.join(t2, t1[c1] == t2[c2]).join(t3, t2[c3] == t3[c4])
    return t3

@timeit
def join_b(t1, t2, t3, c1, c2, c3, c4):
    t3 = t1.join(func.broadcast(t2), t1[c1] == t2[c2]).join(func.broadcast(t3), t2[c3] == t3[c4])
    return t3

@timeit
def group_by(df_joined):   
    return df_joined.groupBy('title').agg(func.mean('rating').alias('avg_rating'),func.count('rating').\
        alias('r_count')).filter('r_count >2').filter('avg_rating > 4.5')


if __name__ == "__main__":
    global sc, spark
    # conf = SparkConf().setMaster('spark://192.168.1.107:7077').setAppName('sparkJoinDemo')
    conf = SparkConf().setMaster('local[4]').setAppName('sparkJoinDemoLocal')
    sc = SparkContext(conf=conf)
    spark = SparkSession.builder.getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    spark.conf.set("spark.executor.memory","16g")
    spark.conf.set("spark.driver.memory","16g")
    spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")

    df_links, df_ratings, df_metas = load_files("links.csv", "ratings.csv","movies_metadata.csv")
    print(df_ratings.rdd.getNumPartitions())
    df_ratings = df_ratings.repartition(10, 'rating')

    # spark.sql("show tables").show()
    # df_links.write.mode("overwrite").saveAsTable("links")
    # df_ratings.write.mode("overwrite").saveAsTable("ratings")
    # df_metas.write.mode("overwrite").saveAsTable("metas")
    # spark.sql("show tables").show()

    # spark.sql("select 'links', count(*) from links").show()
    # spark.sql("select 'ratings', count(*) from ratings").show()
    # spark.sql("select 'metas', count(*) from metas").show()

    df_joined = join(df_ratings, df_links, df_metas, 'movieId','movieId','imdbId','imdb_id')
    group_by(df_joined).show()

    df_joined = join_b(df_ratings, df_links, df_metas, 'movieId','movieId','imdbId','imdb_id')
    group_by(df_joined).show()




