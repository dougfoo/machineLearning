from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import pandas as pd

conf = SparkConf().setMaster('local[3]').setAppName('sparkTest4')
sc = SparkContext(conf=conf)
# spark = SparkSession.builder.master('spark://192.168.1.107:7077').appName("SparkSession3").getOrCreate()

spark = SparkSession.builder.getOrCreate()
# spark.conf.set("spark.sql.execution.arrow.enabled", "true")
# spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")

# spark.sql("show tables").show()
# df_links.write.mode("overwrite").saveAsTable("links")
# df_ratings.write.mode("overwrite").saveAsTable("ratings")
# df_metas.write.mode("overwrite").saveAsTable("metas")
# spark.sql("show tables").show()

# spark.sql("select 'links', count(*) from links").show()
# spark.sql("select 'ratings', count(*) from ratings").show()
# spark.sql("select 'metas', count(*) from metas").show()

# df = spark.createDataFrame([[1],[2]])
# df.write.saveAsTable('footable5')

spark.sql("show databases").show()
spark.sql('show tables').show()

print(sc)
print(spark)

rdd = sc.parallelize([1,2,3,4,5,6,7])
print(rdd.reduce(lambda a, b: a + b))
print(rdd)

