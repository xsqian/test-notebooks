# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
from typing import Tuple

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

import mlrun


@mlrun.handler(labels={"a": 1, "b": "a test", "c": [1, 2, 3]})
def set_labels(arg1, arg2=23):
    return arg1 - arg2


@mlrun.handler(labels={"wrapper_label": "2"})
def set_labels_from_function_and_wrapper(context: mlrun.MLClientCtx = None):
    if context:
        context.set_label("context_label", 1)


@mlrun.handler(
    outputs=[
        "my_array",
        "my_df : dataset",
        "my_dict: dataset",
        "my_list :dataset",
    ]
)
def log_dataset() -> Tuple[np.ndarray, pd.DataFrame, dict, list]:
    return (
        np.ones((10, 20)),
        pd.DataFrame(np.zeros((20, 10))),
        {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]},
        [["A"], ["B"], [""]],
    )


@mlrun.handler(
    outputs=[
        "my_dir: directory",
    ]
)
def log_directory(path: str) -> str:
    path = os.path.join(path, "my_new_dir")
    os.makedirs(path, exist_ok=True)
    open(os.path.join(path, "a.txt"), "a").close()
    open(os.path.join(path, "b.txt"), "a").close()
    open(os.path.join(path, "c.txt"), "a").close()
    return path


@mlrun.handler(
    outputs=[
        "my_file: file",
    ]
)
def log_file(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    my_file = os.path.join(path, "a.txt")
    with open(my_file, "w") as text_file:
        text_file.write("MLRun decorator test.")
    return my_file


@mlrun.handler(outputs=["my_object : object"])
def log_object() -> Pipeline:
    encoder_to_imputer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(missing_values="", strategy="constant", fill_value="C"),
            ),
            ("encoder", OrdinalEncoder()),
        ]
    )
    encoder_to_imputer.fit([["A"], ["B"], ["C"]])
    return encoder_to_imputer


@mlrun.handler(outputs=["my_plot: plot"])
def log_plot() -> plt.Figure:
    my_plot, axes = plt.subplots()
    axes.plot([1, 2, 3, 4])
    return my_plot


@mlrun.handler(
    outputs=[
        "my_int",
        "my_float",
        "my_dict: result",
        "my_array:result",
    ]
)
def log_result() -> Tuple[int, float, dict, np.ndarray]:
    return 1, 1.5, {"a": 1, "b": 2}, np.ones(3)


@mlrun.handler(
    outputs=["my_result", "my_dataset", "my_object", "my_plot", "my_imputer"]
)
def log_as_default_artifact_types():
    my_plot, axes = plt.subplots()
    axes.plot([1, 2, 3, 4])
    return (
        10,
        pd.DataFrame(np.ones(10)),
        cloudpickle.dumps({"a": 5}),
        my_plot,
        SimpleImputer(),
    )


@mlrun.handler(outputs=["dataset: dataset", "result: result", "no_type", None])
def log_with_none_values(
    is_none_dataset: bool = False,
    is_none_result: bool = False,
    is_none_no_type: bool = False,
):
    return (
        None if is_none_dataset else np.zeros(shape=(5, 5)),
        None if is_none_result else 5,
        None if is_none_no_type else np.ones(shape=(10, 10)),
        10,
    )


@mlrun.handler(outputs=["wrapper_dataset: dataset", "wrapper_result: result"])
def log_from_function_and_wrapper(context: mlrun.MLClientCtx = None):
    if context:
        context.log_result(key="context_result", value=1)
        context.log_dataset(key="context_dataset", df=pd.DataFrame(np.arange(10)))
    return [1, 2, 3, 4], "hello"


@mlrun.handler()
def parse_inputs_from_type_hints(
    my_data: list,
    my_encoder: Pipeline,
    data_2,
    data_3: mlrun.DataItem,
    add,
    mul: int = 2,
):
    assert data_2 is None or isinstance(data_2, mlrun.DataItem)
    assert data_3 is None or isinstance(data_3, mlrun.DataItem)

    return (my_encoder.transform(my_data) + add * mul).tolist()


@mlrun.handler(inputs={"my_data": list})
def parse_inputs_from_wrapper_using_types(my_data, my_encoder, add, mul: int = 2):
    if isinstance(my_encoder, mlrun.DataItem):
        my_encoder = my_encoder.local()
        with open(my_encoder, "rb") as pickle_file:
            my_encoder = cloudpickle.load(pickle_file)
    return (my_encoder.transform(my_data) + add * mul).tolist()


@mlrun.handler(
    inputs={
        "my_list": "list",
        "my_array": "numpy.ndarray",
        "my_encoder": "sklearn.pipeline.Pipeline",
    },
    outputs=["result"],
)
def parse_inputs_from_wrapper_using_strings(
    my_list, my_array, my_df, my_encoder, add, mul: int = 2
):
    if isinstance(my_df, mlrun.DataItem):
        my_df = my_df.as_df()
    assert my_list == [["A"], ["B"], [""]]
    assert isinstance(my_encoder, Pipeline)
    return int((my_df.sum().sum() + my_array.sum() + add) * mul)


@mlrun.handler(outputs=["error_numpy"])
def raise_error_while_logging():
    return np.ones(shape=(7, 7, 7))


class MyClass:
    def __init__(self, class_parameter: int):
        assert isinstance(class_parameter, int)
        self._parameter = class_parameter

    @mlrun.handler(
        outputs=[
            "my_array:dataset",
            "my_df : dataset",
            "my_dict :dataset",
            "my_list: dataset",
        ]
    )
    def log_dataset(self) -> Tuple[np.ndarray, pd.DataFrame, dict, list]:
        return (
            np.ones((10, 20)),
            pd.DataFrame(np.zeros((20, 10))),
            {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]},
            [["A"], ["B"], [""]],
        )

    @mlrun.handler(outputs=["my_object: object"])
    def log_object(self) -> Pipeline:
        encoder_to_imputer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        missing_values="", strategy="constant", fill_value="C"
                    ),
                ),
                ("encoder", OrdinalEncoder()),
            ]
        )
        encoder_to_imputer.fit([["A"], ["B"], ["C"]])
        return encoder_to_imputer

    @mlrun.handler(outputs=["result"])
    def parse_inputs_from_type_hints(
        self,
        my_data: list,
        my_encoder: Pipeline,
        data_2,
        data_3: mlrun.DataItem,
        mul: int,
    ):
        assert data_2 is None or isinstance(data_2, mlrun.DataItem)
        assert data_3 is None or isinstance(data_3, mlrun.DataItem)

        return int(sum(my_encoder.transform(my_data) + self._parameter * mul))
