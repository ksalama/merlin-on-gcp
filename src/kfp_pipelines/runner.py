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
"""KFP training pipeline utils."""

from kfp.v2 import compiler
from src.kfp_pipelines import config, training_pipeline
from kfp.v2.google.client import AIPlatformClient


def compile_pipeline(pipeline_definition_path):
    compiler.Compiler().compile(
        pipeline_func=training_pipeline.movielens_training,
        package_path=pipeline_definition_path
    )
    
    
def run_pipeline(pipeline_definition_file, parameter_values):

    client = AIPlatformClient(
        project_id=config.PROJECT,
        region=config.REGION,
    )

    response = client.create_run_from_job_spec(
        pipeline_definition_file,
        parameter_values=parameter_values,
        pipeline_root=config.ARTIFACT_STORE_URI,
        enable_caching=config.ENABLE_CACHING == '1',
        service_account=config.PIPELINES_SA
    )
    
    return response