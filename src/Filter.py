'''
    Filter the Yelp dataset
'''

# Import libraries
import sys
sys.path.insert(0, '.')

import pandas as pd

# pyspark sql
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, struct, lit
from pyspark.sql.types import DoubleType,IntegerType,FloatType

# pyspark machine learning
from pyspark.ml.feature import OneHotEncoder, StringIndexer

# distance
import geopy.distance
from geopy.distance import geodesic

# utils
from commons.Utils import Utils
from spark.utils import *


# Split line by commas
def split_comma(line: str):
    splits = Utils.COMMA_DELIMITER.split(line)
    return "{}, {}".format(splits[1], splits[2])

# Filter the yelp_business dataset by restaurants
def restaurant_filter():

    # init spark session
    ss = spark_init('restaurant_filter')

    # load data
    df = load_df(ss,'../data/yelp_business_numerical.csv')

    df = df.filter(df.categories.like('%Restaurants%'))
    df = df.withColumn('distance', lit(None).cast(FloatType()))
    print_df(df)

    # Clean up
    df = df[['restaurantID', 'name', 'latitude', 'longitude', 'businessID']]

    # save to csv
    print('saving data to csv...')
    df.toPandas().to_csv("../data/yelp_restaurant.csv", encoding='utf-8', index=False)

def calculate_distance(uCords, rCords):
    return geodesic(uCords, rCords ).km

def distance_filter():
    # init spark session
    ss = spark_init('restaurant_filter')

    # load data
    restaurants = load_df(ss,'../data/yelp_restaurant.csv')
    print_df(restaurants)

    # calculate distances
    loc = (45.393220, -73.493000)
    radius = 20.0
    getDistances = udf(lambda row: calculate_distance(loc, (row.latitude,row.longitude)), FloatType())

    # filter distances
    restaurants = restaurants.withColumn(
    "distance", getDistances(struct(restaurants.longitude, restaurants.latitude)))
    restaurants = restaurants.filter(restaurants.distance < radius).sort(restaurants.distance)
    print_df(restaurants)
    reviews.toPandas().to_csv("../data/yelp_local.csv", encoding='utf-8', index=False)
    # Load review data
    reviews = load_df(ss, '../data/yelp_review.csv')

    # Left join review and restaurant data
    reviews = reviews.join(restaurants, reviews.business_id == restaurants.businessID)

    # Clean up
    reviews = reviews[['business_id', 'user_id', 'stars']]
    print_df(reviews)

    # save to csv
    reviews.toPandas().to_csv("../data/yelp_test.csv", encoding='utf-8', index=False)

def __main__():
    distance_filter()

if __name__ == '__main__':
    __main__()
