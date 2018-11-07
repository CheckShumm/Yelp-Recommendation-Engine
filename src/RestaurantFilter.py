import sys
sys.path.insert(0, '.')

import pandas as pd

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, DataFrame
from commons.Utils import Utils

def splitComma(line: str):
    splits = Utils.COMMA_DELIMITER.split(line)
    return "{}, {}".format(splits[1], splits[2])

if __name__ == "__main__":
    # app config
    conf = SparkConf().setAppName("restaurants").setMaster("local[2]")
    sc = SparkContext(conf = conf)

    # initiate session
    ss = SparkSession(sc)

    # load data
    df = ss.read.options(header=True, inferSchema=True).csv('../data/yelp_business.csv')
    print(df.show(5))
    print(df.columns)

    # Filter Data
    df.drop('neighborhood').collect()
    df.filter(df.categories.like('%Restaurants%'))
    print(df.show(5))
    df.toPandas().to_csv("../data/yelp_restaurant.csv")
