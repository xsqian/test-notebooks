from typing import Tuple

import mlrun
import pandas as pd


@mlrun.handler(outputs=["cleaned_data", "num_rows"])
def get_data(dataset: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, int]:
    dataset[label_column] = dataset[label_column].astype("category").cat.codes
    num_rows = dataset.shape[0]
    return dataset, num_rows
