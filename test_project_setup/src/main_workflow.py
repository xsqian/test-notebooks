import mlrun
from kfp import dsl


# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(name="batch-pipeline")
def pipeline(
    dataset: str,
    label_column: str,
    model_name: str,
    test_size: float,
    random_state: int,
):
    # Get current project
    project = mlrun.get_current_project()

    # Ingest the data set
    ingest = mlrun.run_function(
        "get-data",
        inputs={"dataset": dataset},
        params={"label_column": label_column},
        outputs=["cleaned_data"],
    )

    # Train a model
    train = mlrun.run_function(
        "train",
        inputs={"dataset": ingest.outputs["cleaned_data"]},
        params={
            "label_column": label_column,
            "model_name": model_name,
            "test_size": test_size,
            "random_state": random_state,
        },
        outputs=["model"],
    )
