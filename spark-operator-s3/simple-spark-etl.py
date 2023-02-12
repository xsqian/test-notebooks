from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import argparse
import sys
import mlrun

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--source_path',
    help='''Source path of the dataset''',
    required=True)
parser.add_argument('--target_path',
    help='''Target path of the dataset''',
    required=True)

flags = parser.parse_args(sys.argv[1:])
source_path = flags.source_path
target_path = flags.target_path
job_name = 'simple-spark-etl'

#initiate context
context = mlrun.get_or_create_ctx(job_name)


context.logger.info(f'inputs - {source_path} , {target_path}')
context.logger.info(f'Starting {job_name}')

acccesskey = context.get_secret("AWS_ACCESS_KEY_ID")
secretkey = context.get_secret("AWS_SECRET_ACCESS_KEY")


spark = SparkSession.builder \
    .config("spark.hadoop.fs.s3a.bucket.all.committer.magic.enabled", "true") \
    .config("spark.hadoop.fs.s3a.access.key", acccesskey) \
    .config("spark.hadoop.fs.s3a.secret.key", secretkey) \
    .appName(job_name) \
    .getOrCreate()

df = spark.read.load(source_path, format='csv', header='true', inferSchema='true' )

# Remove spaces from column names
renamed_df = df.select([F.col(col).alias(col.replace(' ', '_')) for col in df.columns])

renamed_df.show(3)
renamed_df.repartition(20).write.parquet(target_path)

spark.stop()