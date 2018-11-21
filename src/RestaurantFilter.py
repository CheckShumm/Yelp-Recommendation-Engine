import sys
sys.path.insert(0, '.')

import pandas as pd

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import lit
from pyspark.sql.types import FloatType

from commons.Utils import Utils

def splitComma(line: str):
    splits = Utils.COMMA_DELIMITER.split(line)
    return "{}, {}".format(splits[1], splits[2])

def restaurantFilter():
    # app config
    conf = SparkConf().setAppName("restaurants").setMaster("local[2]")
    sc = SparkContext(conf = conf)

    # initiate session
    ss = SparkSession(sc)

    # load data
    df = ss.read.options(
    header=True, inferSchema=True).csv('../data/yelp_business_numerical.csv')
    print(df.show(5))
    print(df.columns)
    # Filter Data
    # df = df.drop('neighborhood').collect()
    df = df.filter(df.categories.like('%Restaurants%'))
    df = df.withColumn('distance', lit(None).cast(FloatType()))
    print(df.show(5))

    # Clean up
    df = df[['restaurantID', 'name', 'latitude', 'longitude', 'businessID']]

    # save to csv
    df.toPandas().to_csv("../data/yelp_restaurant.csv", encoding='utf-8', index=False)

if __name__ == "__main__":
    restaurantFilter()
