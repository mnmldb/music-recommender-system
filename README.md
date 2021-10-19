# Music Recommender System

## Overview
This project aims to built and evaluate a collaborative-filter based recommender system for a music dataset.

## Dataset
[Million Song Dataset](http://millionsongdataset.com/) (MSD) collected by 
> Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. 
> The Million Song Dataset. In Proceedings of the 12th International Society
> for Music Information Retrieval Conference (ISMIR 2011), 2011.

The MSD consists of one million songs, with metadata (artist, album, year, etc), tags, partial lyrics content, and derived acoustic features.

The user interaction data comes from the [Million Song Dataset Challenge](https://www.kaggle.com/c/msdchallenge)
> McFee, B., Bertin-Mahieux, T., Ellis, D. P., & Lanckriet, G. R. (2012, April).
> The million song dataset challenge. In Proceedings of the 21st International Conference on World Wide Web (pp. 909-916).

The interaction data consists of *implicit feedback*: play count data for approximately one million users.
The interactions have already been partitioned into training, validation, and test sets, as described below.
  - `cf_train.parquet`
  - `cf_validation.parquet`
  - `cf_test.parquet`

Each of these files contains tuples of `(user_id, count, track_id)`, indicating how many times (if any) a user listened to a specific track.
For example, the first few rows of `cf_train.parquet` look as follows:

|    | user_id                                  |   count | track_id           |
|---:|:-----------------------------------------|--------:|:-------------------|
|  0 | b80344d063b5ccb3212f76538f3d9e43d87dca9e |       1 | TRIQAUQ128F42435AD |
|  1 | b80344d063b5ccb3212f76538f3d9e43d87dca9e |       1 | TRIRLYL128F42539D1 |
|  2 | b80344d063b5ccb3212f76538f3d9e43d87dca9e |       2 | TRMHBXZ128F4238406 |
|  3 | b80344d063b5ccb3212f76538f3d9e43d87dca9e |       1 | TRYQMNI128F147C1C7 |
|  4 | b80344d063b5ccb3212f76538f3d9e43d87dca9e |       1 | TRAHZNE128F9341B86 |

## Collaborative-filter Based Recommender System
The collaborative-filter based recommendation model used [Spark's alternating least squares (ALS) method](https://spark.apache.org/docs/2.4.7/ml-collaborative-filtering.html) to learn latent factor representations for users and items.

The model has some hyper-parameters that should be tuned to optimize performance on the validation set, notably: 

  - the *rank* (dimension) of the latent factors, and
  - the regularization parameter.

### Evaluation
The model's accuracy was evaluated on the validation and test data based on predictions of the top 500 items for each user.

The mean Average Precision (mAP) was used as an evaluation metrics (see PySpark's [ranking metrics](https://spark.apache.org/docs/2.4.7/mllib-evaluation-metrics.html#ranking-systems) for more detail).

### Cluster
[The Peel Big Data cluster](https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/peel) in NYU High Performance Computing was used to train the model. The cluster has 18 data nodes, which runs Cloudera CDH version 6.3.4 that includes Hadoop 3.0.0 and Spark 2.4.0.

## Popularity-based Model as Baseline
Additionaly, the popularity-based model was trained as a baseline to compare the performance of the collaborative-filter based model: see https://link.springer.com/chapter/10.1007/978-1-4899-7637-6_3 , section 3.2; or the `bias` implementation provided by [lenskit](https://lkpy.readthedocs.io/en/stable/bias.html) for details.

## Result
Click here to see the [report](https://github.com/mnmldb/music-recommender-system/blob/master/Report.pdf).