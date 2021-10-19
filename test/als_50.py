from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.sql import Window
from pyspark.sql.functions import col, expr
import pyspark.sql.functions as F
from pyspark.mllib.evaluation import RankingMetrics

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("ALS_Draft")\
        .config('spark.blacklist.enabled', False)\
        .getOrCreate()
    
    # file path
    path_train_tiny = 'hdfs:/user/dn2153/cf_train_tiny_new2.parquet'
    path_train_small = 'hdfs:/user/dn2153/cf_train_small_new2.parquet'
    path_train_medium = 'hdfs:/user/dn2153/cf_train_medium_new2.parquet'
    path_train_all = 'hdfs:/user/bm106/pub/MSD/cf_train_new.parquet'
    path_test = 'hdfs:/user/bm106/pub/MSD/cf_test.parquet'

    # read file
    path_train_use = path_train_all

    file_train = spark.read.parquet(path_train_use).rdd
    file_test = spark.read.parquet(path_test).rdd

    # create dataframe
    df_train_raw = spark.createDataFrame(file_train)
    df_test_raw = spark.createDataFrame(file_test)

    # union
    df_all = df_train_raw.union(df_test_raw)

    # StringIndexer
    indexer_user = StringIndexer(inputCol='user_id', outputCol='userIndex').fit(df_all)
    indexer_track = StringIndexer(inputCol='track_id', outputCol='trackIndex').fit(df_all)
    
    # convert
    df_train_stg = indexer_user.transform(df_train_raw)
    df_train = indexer_track.transform(df_train_stg)
    df_test_stg = indexer_user.transform(df_test_raw)
    df_test = indexer_track.transform(df_test_stg)

    # modeling
    als = ALS(rank=20, maxIter=20, regParam=1, alpha=1, implicitPrefs=True, userCol='userIndex', itemCol='trackIndex', ratingCol='count', coldStartStrategy="drop")
    model = als.fit(df_train)

    # Generate top m movie recommendations for a specified set (n) of users
    m = 50
    userRecs = model.recommendForAllUsers(numItems=m)
    # userRecs.show()
    
    # recommendations
    # https://sparkbyexamples.com/pyspark/pyspark-loop-iterate-through-rows-in-dataframe/#use-foreach-loop-through-dataframe
    rdd_pred = userRecs.rdd.map(lambda x: (x.userIndex, [i[0] for i in x.recommendations]))
    df_pred = rdd_pred.toDF(['userIndex', 'trackIndexPred'])
    # df_pred.show()

    # actual
    # https://vinta.ws/code/spark-ml-cookbook-pyspark.html
    windowSpec = Window.partitionBy('userIndex').orderBy(col('count').desc())
    df_true = df_test \
    .select('userIndex', 'trackIndex', F.rank().over(windowSpec).alias('count')) \
    .groupBy('userIndex') \
    .agg(expr('collect_list(trackIndex) as trackIndexTrue')) 
    # df_true.show()

    # join
    rdd_join = df_pred.join(df_true, 'userIndex').rdd.map(lambda x: (x.trackIndexPred, x.trackIndexTrue))
    rankingMetrics = RankingMetrics(rdd_join)
    df_join = rdd_join.toDF(['trackIndexPred', 'trackIndexTrue'])
    # df_join.show()

    # 
    print(rankingMetrics.meanAveragePrecision)

    spark.stop()