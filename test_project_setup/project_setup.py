import os

import mlrun


def setup(project: mlrun.projects.MlrunProject) -> mlrun.projects.MlrunProject:
    source = project.get_param("source")
    secrets_file = project.get_param("secrets_file")
    default_image = project.get_param("default_image")

    if source:
        print(f"Project Source: {source}")
        project.set_source(project.get_param("source"), pull_at_runtime=True)

    if secrets_file and os.path.exists(secrets_file):
        project.set_secrets(file_path=secrets_file)
        mlrun.set_env_from_file(secrets_file)

    if default_image:
        project.set_default_image(default_image)

    # MLRun Functions
    project.set_function(
        name="get-data",
        func="src/data.py",
        kind="job",
        handler="get_data",
    )

    project.set_function(
        name="train",
        func="src/train.py",
        kind="job",
        handler="train_model",
    )

    # MLRun Workflows
    project.set_workflow("main", "src/main_workflow.py")

    # Save and return the project:
    project.save()
    return project
