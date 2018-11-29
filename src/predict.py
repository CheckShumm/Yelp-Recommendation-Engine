import sys
sys.path.insert(0, '.')

import pandas as pd
import json
import math

# SparkSQL imports
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql.functions import max, struct, first
from pyspark.sql.types import DoubleType,IntegerType

# Spark Machine Learning imports
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, CrossValidator
from pyspark.ml.feature import OneHotEncoder, StringIndexer

# utils
from spark.utils import *
from pprint import pprint

def parse_json(user_id):
        with open('../data/recs.json') as f:
            recs = json.load(f)

        conf = SparkConf().setAppName("app").setMaster("local[2]")
        sc = SparkContext(conf = conf)
        ss = SparkSession(sc)

        restaurants = sc.parallelize([(k,)+(v,) for k,v in recs.items()]).toDF(['business_id','stars'])
        user = sc.parallelize([(user_id,) for k in recs.items()]).toDF(['user_id'])
        data = restaurants.join(user).toDF()
        print(user.show(5))
        print(restaurants.show(5))
        print(data.show(5))
        
def predict():
    # init spark session
    ss = spark_init('als')

    # load restaurant review data
    df = load_df(ss,'../data/yelp_numerical.csv')
    df = df.repartition(5)

    # Create training, validation and test dataset 60/20/20
    (training, test) = df.randomSplit([0.8, 0.2])

    # # predict restaurant ratings for users
    model = ALSModel.load("../model")
    # predictions = model.transform(test)
    # evaluator = RegressionEvaluator(metricName="rmse", labelCol="stars",
    #                                 predictionCol="prediction")
    #
    # rmse = evaluator.evaluate(predictions)
    # print("Root-mean-square error = " + str(rmse))
    #
    # # Generate top 10 movie recommendations for each user
    print("Creating recommendation for user")
    user_recs = model.recommendForAllUsers(5)

    # save to csv
    user_recs.toPandas().to_csv("../data/yelp_recs.csv", encoding='utf-8', index=False)

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
    with open('../data/recs.json', 'w') as fp:
        json.dump(recs, fp, sort_keys=True, indent=4)

def __main__():
    parse_json(148)

if __name__ == '__main__':
    __main__()
