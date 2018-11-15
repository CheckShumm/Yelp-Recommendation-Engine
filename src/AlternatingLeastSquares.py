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

def get_recs_for_user(recs):
    #Recs should be for a specific user.
    recs = recs.select("recommendations.business_id", "recommendations.stars")
    restaurants = recs.select("business_id").toPandas().iloc[0,0]
    stars = recs.select("stars").toPandas.iloc[0,0]
    ratings_matrix = pd.DataFrame(restaurants, columns = ["business_id"])
    ratings_matrix["stars"] = stars
    ratings_matrix_ps = sqlContext.createDataFrame(ratings_matrix)
    return ratings_matrix_ps

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
    restaurantReviews = restaurantReviews.repartition(5)

    print(restaurantReviews.show(5))
    print("number of partitions: ", restaurantReviews.rdd.getNumPartitions())
    print("number of rows: ", restaurantReviews.count())
    print("Creating training and testing dataset")

    # Create training, validation and test dataset 60/20/20
    (training, test) = restaurantReviews.randomSplit([0.8, 0.2])

    als = ALS(maxIter=5, regParam=0.01, userCol="userID", itemCol="restaurantID", ratingCol="stars",
          coldStartStrategy="drop")

    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="stars",
                                    predictionCol="prediction")

    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

    # Generate top 10 movie recommendations for each user
    userRecs = model.recommendForAllUsers(10)
    print(userRecs.show(5))
    # print("Creating and Tuning ALS Model")
    # # Create Alternating Least Squares Model
    # als = ALS(userCol="userID", itemCol="restaurantID", ratingCol="stars",
    # coldStartStrategy="drop", nonnegative=True)
    #
    # # Tune model using ParamGridBuilder
    # param_grid = ParamGridBuilder()\
    #             .addGrid(als.rank, [12,13,14])\
    #             .addGrid(als.maxIter, [18, 19, 20])\
    #             .addGrid(als.regParam, [.17, .18, .19])\
    #             .build()
    #
    # # Define evaluator as RMSE
    # evaluator = RegressionEvaluator(metricName="rmse", labelCol="stars",
    #                                 predictionCol="prediction")
    #
    # print("Create Train Validation Split")
    # # Build cross validation using TrainValidationSplit
    # cv = CrossValidator(estimator=als,
    #                     estimatorParamMaps=param_grid,
    #                     evaluator=evaluator,
    #                     numFolds=3)
    #
    # print("Fitting Model")
    # Fit model to training Data
    # model = cv.fit(training)
    #
    # # Extract best model from tuning
    # best_model = model.bestModel
    #
    # # Generate predictions and evaluate using RMSE
    # predictions = best_model.transform(test)
    # rmse = evvaluator.evaluate(predictions)
    #
    # print(predictions.sort("user_id", "stars").show(5))
    # # Print evaluation metrics and model parameters
    # print ("RMSE = ", rmse)
    # print("*** Best Model ***")
    # print("  Rank: ", best_model.rank)
    # print("  MaxIter:", best_model._java_obj.parent().getMaxIter())
    # print("  RegParam:", best_model._java_obj.parent().getRegParam())
    #
    # # get recommendation for all users
    # userRecs  = best_model.recommendForAllUsers(10)
    # userRecsDF = get_recs_for_user(userRecs)
