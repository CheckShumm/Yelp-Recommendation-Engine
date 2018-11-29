import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
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

# for post request
import requests

# utils
from spark.utils import *

import functools

def unionAll(dfs):
    return functools.reduce(lambda df1,df2: df1.union(df2.select(df1.columns)), dfs)

def parse_json():
        with open('../data/data.json') as f:
            data = json.load(f)
        df = sc.parallelize([(r['business_id'],)+(data['user_id'],)+(r['stars'],) for r in data["reviews"]]).toDF(['business_id','user_id','stars'])
        return df

def als():

    # app config
    conf = SparkConf().setAppName("als").setMaster("local[*]")
    sc = SparkContext(conf = conf)
    ss = SparkSession(sc)

    # load restaurant review data
    path = '../data/yelp_test.csv'
    df = ss.read.options(
    header=True, inferSchema=True).csv(path)

    print('Intiating Spark session.')

    # manually importing json for now
    with open('../data/data.json') as f:
        data = json.load(f)
    user_id = data['user_id']
    new_user = sc.parallelize([(r['business_id'],)+(data['user_id'],)+(r['stars'],) for r in data["reviews"]]).toDF(['business_id','user_id','stars'])
    new_user = new_user.withColumn("stars", new_user['stars'].cast(IntegerType()))
    df = unionAll([new_user, df])
    #print_df(new_user)
    #print_df(df)
    df = df.repartition(5)


    # print("number of partitions: ", df.rdd.getNumPartitions())
    # print_df(df)

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
    #model.save("../model")

    print("Testing model")
    Evaluate the model by computing the RMSE on the test data
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
    int_user_id = df.where(df.user_id == "1").select('int_user_id').collect()[0].__getitem__("int_user_id")
    print("INT USER ID: ", int_user_id)
    recommendations = (user_recs.where(user_recs.int_user_id == int_user_id).select('recommendations')).head()[0]

    # Convert to dictionary
    for row in recommendations:
        #print(row)
        id = row.__getitem__("int_business_id")
        rating = row.__getitem__("rating")
        business_id = df.where(df.int_business_id == id).select('business_id').collect()[0]
        recs[business_id[0]] = rating

    # save as json
    with open('recs.json', 'w') as fp:
        json.dump(recs, fp, sort_keys=True, indent=4)

    url = "http://35.182.248.84/api/recommend"

    querystring = {"user_id":user_id}

    payload = recs
    headers = {
        'Content-Type': "application/json",
        'Authorization': "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImp0aSI6ImUxZGVlNjM1M2ZjM2E0ZjRiMmFkMTNkZGZlMWFmNGQ2NGJkYTFjOGU2Y2JmYjYzMTc0MDJlMGExZDI3MzkwYjc0ZjU1ODljYjM4YTNmMjI4In0.eyJhdWQiOiI1IiwianRpIjoiZTFkZWU2MzUzZmMzYTRmNGIyYWQxM2RkZmUxYWY0ZDY0YmRhMWM4ZTZjYmZiNjMxNzQwMmUwYTFkMjczOTBiNzRmNTU4OWNiMzhhM2YyMjgiLCJpYXQiOjE1NDMzODMwOTUsIm5iZiI6MTU0MzM4MzA5NSwiZXhwIjoxNTc0OTE5MDk1LCJzdWIiOiI2Iiwic2NvcGVzIjpbXX0.ZGu3d4MX7zTo01YbQ7YdnooYojRJOdecdxNJcdEMi9IwQ03zLr0OPlpeQ0DtYFi6tZ-3NrlxtaPMfuZ_EpAaiYe8VI1-NkPL7_zjBGbdtYt1BXVCGEeWvrKrsPXlICxvIcWG00TeWTzoPUw8LH_qNgwHU_4hgx8RsjN7CFongwybMcX0tm4mdgbyQ0TMBzjZDbZ_8gJSMB-9J-XkqjZvx4lEolzlIwTQsawobi7DVgL9dvCncHgMcksBwQJav_DEsiatJX_-LOgt9uRWAg-QVwNuHItehil1vheAWHTJ5UOm6QkEVwvrIYr4zC1BVOLL3W7NCg-Eb6NFm3HYUYuO1fVukzXcOS_b8NSn0JTJ2cCuPYMd3vy90RlVmyxEVyf8AsVijg_j2o-JOCv4BcXxvZ7MvthZ-ElPZEVOvX8azqWFo_5pFN2zw4on1sjqIp_qM5JNUvB8UUOk_bzoVZnIqiDn5YCVplroDY-yElwzTL2dirFV1aducqIoKLbIud1cAONT_N1IfGoox7Vru-D0Va4ksCfnMS8-N6OQPCoq_tiFlvMVcM6sAVnYqgqD7-h76xrmmmFxTbEC8JJpRkSLQm-aeqqy5mrdXFmlJfCibaZRagrXEzklIxXkjFax_Tug0PSf-u1kOZs99Uzt9aIRoUoEO6PVd48Jf0mxd47avQY",
        'cache-control': "no-cache",
        'Postman-Token': "06d22cd5-aad5-4d51-ba9a-8a8cc81afb1c"
    }

    response = requests.request("POST", url, data=payload, headers=headers, params=querystring)

    print(response.text)

def __main__():
    als()

if __name__ == '__main__':
    __main__()
