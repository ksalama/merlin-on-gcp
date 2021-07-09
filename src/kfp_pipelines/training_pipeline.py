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
"""KFP training pipeline."""

import os
import json
import kfp.v2.dsl as dsl

from src.kfp_pipelines import config
from src.kfp_pipelines import components

@dsl.pipeline(name=config.PIPELINE_NAME)
def movielens_training(
    num_epochs: int=1,
    learning_rate: float=0.001,
    batch_size: int=10240
):
    
    get_data = components.get_data_op(
        project=config.PROJECT,
        region=config.REGION,
        movies_dataset_display_name=config.MOVIES_DATASET_DISPLAY_NAME,
        ratings_dataset_display_name=config.RATINGS_DATASET_DISPLAY_NAME,
    )
    
    prep_data = components.data_etl_op(
        project=config.PROJECT,
        region=config.REGION,
        staging_location=os.path.join(config.GCS_LOCATION, "jobs"), 
        service_account=config.VERTEX_SERVICE_ACCOUNT,
        tensorboard_name=config.TENSORBOARD_RESOURCE_NAME,
        vertex_training_machine_spec=json.dumps(config.VERTEX_TRAINING_MACHINE_SPEC),
        image_uri=config.NVT_IMAGE_URI,
        dataset=get_data.outputs['dataset']
    )
    
    train_model = components.train_op(
        project=config.PROJECT,
        region=config.REGION,
        staging_location=os.path.join(config.GCS_LOCATION, "jobs"), 
        service_account=config.VERTEX_SERVICE_ACCOUNT,
        tensorboard_name=config.TENSORBOARD_RESOURCE_NAME,
        vertex_training_machine_spec=json.dumps(config.VERTEX_TRAINING_MACHINE_SPEC),
        image_uri=config.NVT_IMAGE_URI,
        num_epochs=num_epochs,
        batch_size=batch_size, 
        learning_rate=learning_rate,
        etl_output=prep_data.outputs['etl_output']
    )
    
    upload_model = components.upload_model_op(
        project=config.PROJECT,
        region=config.REGION,
        model_display_name=config.MODEL_DISPLAY_NAME,
        serving_container_image_uri=config.SERVING_IMAGE_URI,
        model=train_model.outputs['model']
    )