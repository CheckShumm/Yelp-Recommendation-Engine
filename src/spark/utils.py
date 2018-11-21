'''
    Spark Utils
'''

# import libraries
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, DataFrame

# initialize spark session
def spark_init(app):
    print('Intiating Spark session.')

    # app config
    conf = SparkConf().setAppName("app").setMaster("local[2]")
    sc = SparkContext(conf = conf)

    # initiate session
    return SparkSession(sc)

# Read csv to dataframe
def load_df(ss, path):
    # load data
    print('Loading data ...')

    return ss.read.options(
    header=True, inferSchema=True).csv(path)

# print df details
def print_df(df):
    print("number of rows: ", df.count())
    print(df.show(5))
    print("Column Types:", df.dtypes)
