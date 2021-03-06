import sys
sys.path.insert(0, '.')

import pandas as pd
import math

# SparkSQL imports
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, struct, lit
from pyspark.sql.types import DoubleType,IntegerType,FloatType
from pyspark.ml.feature import OneHotEncoder, StringIndexer
# distance
import geopy.distance
from geopy.distance import geodesic

def calculateDistance(uCords, rCords):
    return geodesic(uCords, rCords ).km

def distanceFilter():
        # app config
        conf = SparkConf().setAppName("restaurants").setMaster("local[*]")
        sc = SparkContext(conf = conf)

        # initiate session
        ss = SparkSession(sc)

        # load data
        restaurants = ss.read.options(
        header=True, inferSchema=True).csv("../data/yelp_restaurant.csv")

        print(restaurants.show(5))
        #print(restaurants.filter(restaurants.latitude < 5000.0).count())
        mtl = (45.393220, -73.493000)

        getDistances = udf(lambda row: calculateDistance(mtl, (row.latitude,row.longitude)), FloatType())
        restaurants = restaurants.withColumn("distance", getDistances(struct(restaurants.longitude, restaurants.latitude)))

        print(restaurants.dtypes)
        print(restaurants.show(5))
        print("number of rows: ", restaurants.count())
        radius = 5.0
        restaurants = restaurants.filter(restaurants.distance < radius).sort(restaurants.distance)
        print("number of rows: ", restaurants.count())
        # save to csv
        #restaurants.toPandas().to_csv("../data/yelp_local.csv", encoding='utf-8', index=False)

        # # load data
        # print("opening reviews...")
        # df = ss.read.options(
        # header=True, inferSchema=True).csv("../data/yelp_review.csv")
        #
        # # convert User IDs to numerical value
        # indexer = StringIndexer(inputCol="user_id", outputCol="userID", handleInvalid="skip")
        # df = indexer.fit(df).transform(df)
        # df = df.join(restaurants, df.business_id == restaurants.businessID)
        #
        # # Clean up
        # df = df[['restaurantID','name', 'userID', 'stars']]
        # print(df.show(5))
        # print(df.dtypes)
        # print("number of rows: ", df.count())
        # # save to csv
        # df.toPandas().to_csv("../data/yelp_test.csv", encoding='utf-8', index=False)


def filterLocalReviews():
        # app config
        conf = SparkConf().setAppName("restaurants").setMaster("local[*]")
        sc = SparkContext(conf = conf)

        # initiate session
        ss = SparkSession(sc)

        # load data
        print("opening reviews...")
        df = ss.read.options(
        header=True, inferSchema=True).csv("../data/yelp_review.csv").limit(10000)
        indexer = StringIndexer(inputCol="user_id", outputCol="userID")
        df = indexer.fit(df).transform(df)
        df = df.join(restaurants, df.business_id == restaurants.businessID)

        # Clean up
        df = df[['restaurantID','name', 'userID', 'stars']]
        print(df.show(5))
        print(df.dtypes)

        # save to csv
        #df.toPandas().to_csv("../data/yelp_test.csv", encoding='utf-8', index=False)

if __name__ == "__main__":
    distanceFilter()
