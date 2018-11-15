import sys
sys.path.insert(0, '.')

import pandas as pd
import math

# SparkSQL imports
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, struct, lit
from pyspark.sql.types import DoubleType,IntegerType,FloatType

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
        radius = 20
        getDistances = udf(lambda row: calculateDistance(mtl, (row.latitude,row.longitude)), FloatType())
        restaurants = restaurants.withColumn("distance", getDistances(struct(restaurants.longitude, restaurants.latitude)))

        print(restaurants.dtypes)

        print(restaurants.show(5))
        radius = 5000.0
        restaurants = restaurants.filter(restaurants.distance < 5000.0).sort(restaurants.distance)
        print(restaurants.show(5))

        # save to csv
        restaurants.toPandas().to_csv("../data/yelp_local.csv", encoding='utf-8', index=False)

if __name__ == "__main__":
    distanceFilter()
