from pyspark.sql import SparkSession
from pyspark.sql import Row

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("Sampling")\
        .getOrCreate()
    
    # original file path
    path_train = 'hdfs:/user/bm106/pub/MSD/cf_train_new.parquet'
    path_val = 'hdfs:/user/bm106/pub/MSD/cf_validation.parquet'
    path_test = 'hdfs:/user/bm106/pub/MSD/cf_test.parquet'

    # read file
    file_train = spark.read.parquet(path_train).rdd
    file_val = spark.read.parquet(path_val).rdd
    df_train = spark.createDataFrame(file_train)
    df_val = spark.createDataFrame(file_val)

    # files path
    path_sample_tiny = 'hdfs:/user/dn2153/cf_train_tiny.parquet'
    path_sample_small = 'hdfs:/user/dn2153/cf_train_small.parquet'
    path_sample_medium = 'hdfs:/user/dn2153/cf_train_medium.parquet'

    # view
    df_train.createOrReplaceTempView('df_train')
    df_val.createOrReplaceTempView('df_val')

    # sql
    user_id_val = spark.sql('select tr.* \
                            from df_train tr \
                            inner join df_val va \
                            on va.user_id = tr.user_id')
    
    user_id_val_not = spark.sql('select tr.* \
                                from df_train tr \
                                where tr.user_id not in (select distinct user_id from df_val)')

    print(user_id_val.count())
    print(user_id_val_not.count())
    
    # export
    user_id_val.union(user_id_val_not.sample(fraction=0.01, seed=3)).repartition(1).write.parquet(path_sample_tiny)
    user_id_val.union(user_id_val_not.sample(fraction=0.05, seed=3)).repartition(1).write.parquet(path_sample_small)
    user_id_val.union(user_id_val_not.sample(fraction=0.25, seed=3)).repartition(1).write.parquet(path_sample_medium)

    spark.stop()