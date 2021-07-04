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
"""KFP pipeline configurations."""


import os

PROJECT = os.getenv("PROJECT", "merlin-on-gcp")
REGION = os.getenv("REGION", "us-central1")
GCS_LOCATION = os.getenv("GCS_LOCATION", "gs://merlin-on-gcp/movielens25m/")

ARTIFACT_STORE_URI = os.path.join(GCS_LOCATION, "kfp_artifacts")
MODEL_REGISTRY_URI = os.getenv(
    "MODEL_REGISTRY_URI",
    os.path.join(GCS_LOCATION, "model_registry"),
)

MOVIES_DATASET_DISPLAY_NAME = os.getenv("MOVIES_DATASET_DISPLAY_NAME", 'movielens25m-movies')
RATINGS_DATASET_DISPLAY_NAME = os.getenv("MOVIES_DATASET_DISPLAY_NAME", 'movielens25m-ratings')

MODEL_DISPLAY_NAME = os.getenv(
    "MODEL_DISPLAY_NAME", "movielens25m-recommender"
)
PIPELINE_NAME = os.getenv("PIPELINE_NAME", f"{MODEL_DISPLAY_NAME}-train-pipeline")

IMAGE_URI = os.getenv(
    "TFX_IMAGE_URI", f"gcr.io/{PROJECT}/movielens-cuda11.0-tf2.4:latest"
)

VERTEX_SERVICE_ACCOUNT = os.getenv(
    "VERTEX_SERVICE_ACCOUNT", 
    f'vertex-sa-mlops@{PROJECT}.iam.gserviceaccount.com')
PIPELINES_SA = os.getenv("PIPELINES_SA", f'vertex-sa-mlops@{PROJECT}.iam.gserviceaccount.com')
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "1")


VERTEX_TRAINING_MACHINE_SPEC = { 
    "machine_type": "n1-standard-4",
    "accelerator_type": "NVIDIA_TESLA_V100",
    "accelerator_count": 1,
}

TENSORBOARD_RESOURCE_NAME = os.getenv(
    "TENSORBOARD_RESOURCE_NAME",
    "projects/659831510405/locations/us-central1/tensorboards/4450717516120981504"
)


os.environ["PROJECT"] = PROJECT
os.environ["PIPELINE_NAME"] = PIPELINE_NAME
os.environ["IMAGE_URI"] = IMAGE_URI
os.environ["MODEL_REGISTRY_URI"] = MODEL_REGISTRY_URI