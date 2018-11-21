import sys
sys.path.insert(0, '.')

import pandas as pd
import json
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

# utils
from spark.utils import *

def als():
    # init spark session
    ss = spark_init('als')

    # load restaurant review data
    df = load_df(ss,'../data/yelp_test.csv')
    df = df.repartition(5)


    print("number of partitions: ", df.rdd.getNumPartitions())
    print_df(df)

    # Convert user_id and business_id to numerical ID
    user_indexer = StringIndexer(inputCol="user_id", outputCol="int_user_id", handleInvalid="skip")
    df = user_indexer.fit(df).transform(df)

    # Convert user_id and business_id to numerical ID
    business_indexer = StringIndexer(inputCol="business_id", outputCol="int_business_id", handleInvalid="skip")
    df = business_indexer.fit(df).transform(df)

    print("Creating training and testing dataset")

    # Create training, validation and test dataset 60/20/20
    (training, test) = df.randomSplit([0.8, 0.2])


    print("Training model")
    als = ALS(maxIter=5, regParam=0.01, userCol="int_user_id", itemCol="int_business_id", ratingCol="stars",
          coldStartStrategy="drop")

    model = als.fit(training)

    print("Testing model")
    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="stars",
                                    predictionCol="prediction")

    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

    # Generate top 10 movie recommendations for each user
    print("Creating recommendation for user")
    user_recs = model.recommendForAllUsers(5)
    print(user_recs.show(5))

    # Convert int ids to original string ids
    recs = {}
    user_id = df.where(df.int_user_id == 148).select('user_id').collect()[0]
    recommendations = (user_recs.where(user_recs.int_user_id == 148).select('recommendations')).head()[0]

    # Convert to dictionary
    for row in recommendations:
        print(row)
        id = row.__getitem__("int_business_id")
        rating = row.__getitem__("rating")
        business_id = df.where(df.int_business_id == id).select('business_id').collect()[0]
        recs[business_id[0]] = rating

    print(recs)
    print('Recommendations for: ', user_id[0])
    print(recommendations)

    # save as json
    with open('recs.json', 'w') as fp:
        json.dump(recs, fp, sort_keys=True, indent=4)

def __main__():
    als()

if __name__ == '__main__':
    __main__()
