import pandas as pd
import mlrun
from mlrun.artifacts import Artifact


def fetch_data(context, dataset: mlrun.DataItem,format="csv"):
    df  = dataset.as_df()
    context.logger.info("saving dataframe to s3")
    context.log_dataset("dataset", df=df, format=format, index=False)
    