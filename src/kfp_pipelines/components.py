# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KFP custom python components."""

import kfp.v2.dsl as dsl
from kfp.v2.dsl import Artifact, Dataset, Output, Input, Model

@dsl.component(
    base_image='gcr.io/deeplearning-platform-release/base-cpu',
     packages_to_install=['google-cloud-aiplatform']
)
def get_data_op(
    project: str,
    region: str,
    movies_dataset_display_name: str,
    ratings_dataset_display_name: str,
    dataset: Output[Dataset]
):
    
    from google.cloud import aiplatform as vertex_ai
    vertex_ai.init(project=project, location=region)
    
    def _get_dataset_gcs_location(dataset_display_name):
        datasets = vertex_ai.TabularDataset.list()
        dataset = None
        for entry in datasets:
            if entry.display_name == dataset_display_name:
                dataset = entry
                break

        if not dataset:
            raise ValueError(f"Dataset with display name {dataset_display_name} does not exist!")

        return dataset.gca_resource.metadata['inputConfig']['gcsSource']['uri'][0]
    
    dataset.metadata['movies_csv_data_location'] = _get_dataset_gcs_location(movies_dataset_display_name)
    dataset.metadata['ratings_csv_data_location'] = _get_dataset_gcs_location(ratings_dataset_display_name)
    
    

@dsl.component(
    base_image='gcr.io/deeplearning-platform-release/base-cpu',
    packages_to_install=['google-cloud-aiplatform']
)
def data_etl_op(
    project: str,
    region: str,
    staging_location: str, 
    service_account: str,
    tensorboard_name: str,
    vertex_training_machine_spec: str,
    image_uri: str,
    dataset: Input[Dataset],
    etl_output: Output[Artifact]
):
    
    import os
    import json
    import time
    from google.cloud import aiplatform as vertex_ai
    vertex_ai.init(project=project, location=region)
    
    movies_csv_dataset_location = dataset.metadata['movies_csv_data_location']
    ratings_csv_dataset_location = dataset.metadata['ratings_csv_data_location']
    etl_output_dir = etl_output.path.replace("/gcs/", "gs://")
    
    worker_pool_specs =  [
        {
            "machine_spec": json.loads(vertex_training_machine_spec),
            "replica_count": 1,
            "container_spec": {
                "image_uri": image_uri,
                "args": [
                    "python",
                    "-m",
                    "src.data_preprocessing.task",
                    f'--movies-csv-data-location={movies_csv_dataset_location}',
                    f'--ratings-csv-data-location={ratings_csv_dataset_location}',
                    f'--etl-output-dir={etl_output_dir}',
                ],
            },
        }
    ]
    
    job_name = "movielens-nvt-etl-{}".format(time.strftime("%Y%m%d_%H%M%S"))

    job = vertex_ai.CustomJob(
        display_name=job_name,
        worker_pool_specs=worker_pool_specs,
        staging_bucket=staging_location,
    )

    job.run(
        sync=True, 
        service_account=service_account,
        tensorboard=tensorboard_name
    )
    
    etl_output.metadata['transformed_train_data_dir'] = os.path.join(etl_output.path, "transformed_data/train")
    etl_output.metadata['transformed_test_data_dir'] = os.path.join(etl_output.path, "transformed_data/test")
    etl_output.metadata['transform_workflow_dir'] = os.path.join(etl_output.path, "transform_workflow")
    
    
@dsl.component(
    base_image='gcr.io/deeplearning-platform-release/base-cpu',
    packages_to_install=['google-cloud-aiplatform']
)
def train_op(
    project: str,
    region: str,
    staging_location: str, 
    service_account: str,
    tensorboard_name: str,
    vertex_training_machine_spec: str,
    image_uri: str,
    num_epochs: int,
    batch_size: int, 
    learning_rate: float,
    etl_output: Input[Artifact],
    model: Output[Model]
):
    
    import os
    import json
    import time
    from google.cloud import aiplatform as vertex_ai

    vertex_ai.init(project=project, location=region)
    
    transformed_train_data = os.path.join(etl_output.metadata['transformed_train_data_dir'], '*.parquet')
    transformed_eval_data = os.path.join(etl_output.metadata['transformed_test_data_dir'], '*.parquet')
    transform_workflow_dir = etl_output.metadata['transform_workflow_dir']
    model_dir = model.path.replace("/gcs/", "gs://")
    
    worker_pool_specs =  [
        {
            "machine_spec": json.loads(vertex_training_machine_spec),
            "replica_count": 1,
            "container_spec": {
                "image_uri": image_uri,
                "args": [
                    "python",
                    "-m",
                    "src.model_training.task",
                    f'--model-dir={model_dir}',
                    f'--train-data-file-pattern={transformed_train_data}',
                    f'--eval-data-file-pattern={transformed_eval_data}',
                    f'--nvt-workflow-dir={transform_workflow_dir}',
                    f'--num-epochs={num_epochs}',
                    f'--learning-rate={learning_rate}',
                    f'--batch-size={batch_size}',
                ],
            },
        }
    ]
    
    job_name = "movielens-tf-training-{}".format(time.strftime("%Y%m%d_%H%M%S"))

    job = vertex_ai.CustomJob(
        display_name=job_name,
        worker_pool_specs=worker_pool_specs,
        staging_bucket=staging_location,
    )

    job.run(
        sync=True, 
        service_account=service_account,
        tensorboard=tensorboard_name
    )
    
    
@dsl.component(
    base_image='gcr.io/deeplearning-platform-release/base-cpu',
    packages_to_install=['google-cloud-aiplatform']
)
def upload_model_op(
    project: str,
    region: str,
    model_display_name: str,
    serving_container_image_uri: str,
    model: Input[Model],
    uploaded_model: Output[Artifact]
):
    
    import os
    from google.cloud import aiplatform as vertex_ai

    vertex_ai.init(project=project, location=region)
    
    artifact_uri = model.path.replace("/gcs/", "gs://")
    
    vertex_model = vertex_ai.Model.upload(
        display_name=model_display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri
    )
    
    uploaded_model.metadata["model_gca_resource"] = vertex_model.gca_resource
    
    
    