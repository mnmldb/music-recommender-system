from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.sql import Window
from pyspark.sql.functions import col, expr
import pyspark.sql.functions as F
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.types import DoubleType

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("Popularity")\
        .config('spark.blacklist.enabled', False)\
        .getOrCreate()
    
    # file path
    path_train_tiny = 'hdfs:/user/dn2153/cf_train_tiny.parquet'
    path_train_small = 'hdfs:/user/dn2153/cf_train_small.parquet'
    path_train_medium = 'hdfs:/user/dn2153/cf_train_medium.parquet'
    path_train_all = 'hdfs:/user/bm106/pub/MSD/cf_train_new.parquet'
    path_val = 'hdfs:/user/bm106/pub/MSD/cf_validation.parquet'
    path_test = 'hdfs:/user/bm106/pub/MSD/cf_test.parquet'

    # read file
    path_train_use = path_train_all
    path_val_test_use = path_val # choose from the validation set and the test se

    file_train = spark.read.parquet(path_train_use).rdd
    file_val = spark.read.parquet(path_val_test_use).rdd

    # create dataframe
    df_train_raw = spark.createDataFrame(file_train)
    df_val_raw = spark.createDataFrame(file_val)

    # union
    df_all = df_train_raw.union(df_val_raw)

    # StringIndexer
    indexer_user = StringIndexer(inputCol='user_id', outputCol='userIndex').fit(df_all)
    indexer_track = StringIndexer(inputCol='track_id', outputCol='trackIndex').fit(df_all)
    
    # convert
    df_train_stg = indexer_user.transform(df_train_raw)
    df_train = indexer_track.transform(df_train_stg)
    df_val_stg = indexer_user.transform(df_val_raw)
    df_val = indexer_track.transform(df_val_stg)

    # unique users in validation
    unique_user = df_val.select('userIndex').distinct()

    # true interactions
    windowSpec = Window.partitionBy('userIndex').orderBy(col('count').desc())
    df_true = df_val \
    .select('userIndex', 'trackIndex', F.rank().over(windowSpec).alias('count')) \
    .groupBy('userIndex') \
    .agg(expr('collect_list(trackIndex) as trackIndexTrue')) 

    # parameters
    bg = [1, 10, 20, 50] # dumping factor
    bi = [1, 10, 20, 50]
    bu = 1

    # all interactions
    sig = df_train.select('count').groupBy().sum().collect()[0][0]

    # R
    R = df_train.count()

    # grid search
    for i in bg:
        for j in bi:
            
            # global average
            mu = sig / (R + i)

            # track average
            track_num = df_train.groupby('trackIndex').sum('count').withColumnRenamed('sum(count)', 'counts') # Sigma R[u,i]
            track_den = df_train.groupby('trackIndex').count().withColumnRenamed('count', 'interactions') # |R[:,i]|
            track_join = track_num.join(track_den, 'trackIndex')
            
            rdd_track = track_join.rdd.map(lambda x: (x.trackIndex, (x.counts - mu * x.interactions) / (x.interactions + j))) # need to multiply x.interactions
            track = rdd_track.toDF(['trackIndex', 'track_average'])
            
            # user average
            df_train_join = df_train.join(track, 'trackIndex')
            df_train_join = df_train_join.withColumnRenamed('count', 'counts')

            rdd_user_base = df_train_join.rdd.map(lambda x: (x.userIndex, x.trackIndex, x.counts, x.counts - mu - x.track_average))
            user_base = rdd_user_base.toDF(['userIndex', 'trackIndex', 'counts', 'difference'])

            user_num = user_base.groupby('userIndex').sum('difference').withColumnRenamed('sum(difference)', 'difference') # Sigma R[u,i]
            user_den = user_base.groupby('userIndex').count().withColumnRenamed('count', 'interactions') # |R[u]
            user_join = user_num.join(user_den, 'userIndex')

            rdd_user = user_join.rdd.map(lambda x: (x.userIndex, x.difference / (x.interactions + bu))) # difference includes mu and track average 
            user = rdd_user.toDF(['userIndex', 'user_average'])

            # recommendations
            m = 500
            rdd_pred = track.sort(track.track_average.desc()).select('trackIndex').rdd.flatMap(lambda x:x)
            list_pred = rdd_pred.collect()
            list_pred = list_pred[:m]

            rdd_pred = unique_user.rdd.map(lambda x: (x.userIndex, list_pred))
            df_pred = rdd_pred.toDF(['userIndex', 'trackIndexPred'])

            # join
            rdd_join = df_pred.join(df_true, 'userIndex').rdd.map(lambda x: (x.trackIndexPred, x.trackIndexTrue))
            rankingMetrics = RankingMetrics(rdd_join)
            df_join = rdd_join.toDF(['trackIndexPred', 'trackIndexTrue'])
            
            # ranking metrics 
            print(i, j, rankingMetrics.meanAveragePrecision)

    spark.stop()