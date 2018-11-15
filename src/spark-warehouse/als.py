import sys
sys.path.insert(0, '.')

import pandas as pd
import math

# SparkSQL imports
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import max, struct, first
from pyspark.sql.types import DoubleType,IntegerType

# Spark Machine Learning imports
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, CrossValidator
from pyspark.ml.feature import OneHotEncoder, StringIndexer

import pandas as pd

if __name__ == "__main__":
        # app config, local[*] for max available cores
        conf = SparkConf().setAppName("dataPivot").setMaster("local[*]")
        conf.set('spark.sql.pivotMaxValues', u'3230000')
        sc = SparkContext(conf = conf)

        # set pivot max values to restaurant column max
        # initiate session
        ss = SparkSession(sc)

        # load data
        restaurantReviews = ss.read.options(
        header=True, inferSchema=True).csv("../data/yelp_local.csv")
        
