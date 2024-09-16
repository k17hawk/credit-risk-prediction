from pyspark.sql import SparkSession
import os

spark_session = SparkSession.builder.master('local[*]').config("spark.driver.memory", "15g").appName('credit-risk-prediction').getOrCreate()
