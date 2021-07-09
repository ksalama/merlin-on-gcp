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
"""The entrypoint for the Vertex AI etl job."""

import os
import sys
from datetime import datetime
import logging
import nvtabular as nvt
import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse

# from google.cloud import aiplatform as vertex_ai

from src.data_preprocessing import etl
from src.common import features, utils

LOCAL_TRANSFORM_DIR = "transform_workflow"


def get_args():
    parser = argparse.ArgumentParser()

    #     parser.add_argument(
    #         "--movies-dataset-display-name",
    #         type=str,
    #     )

    #     parser.add_argument(
    #         "--ratings-dataset-display-name",
    #         type=str,
    #     )

    parser.add_argument(
        "--movies-csv-data-location",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--ratings-csv-data-location",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--etl-output-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )

    #     parser.add_argument(
    #         "--project",
    #         type=str
    #     )

    #     parser.add_argument(
    #         "--region",
    #         type=str
    #     )

    return parser.parse_args()


def get_dataset_gcs_location(dataset_display_name):
    datasets = vertex_ai.TabularDataset.list()
    dataset = None
    for entry in datasets:
        if entry.display_name == dataset_display_name:
            dataset = entry
            break

    if not dataset:
        raise ValueError(
            f"Dataset with display name {dataset_display_name} does not exist!"
        )

    return dataset.gca_resource.metadata["inputConfig"]["gcsSource"]["uri"][0]


def main():
    args = get_args()

    #     vertex_ai.init(
    #         project=args.project,
    #         location=args.region)

    #     logging.info("Getting GCS data locations...")
    #     movies_csv_data_location = get_dataset_gcs_location(movies_dataset_display_name)
    #     ratings_csv_data_location = get_dataset_gcs_location(ratings_dataset_display_name)

    (
        transformed_train_dataset,
        transformed_test_dataset,
        transform_workflow,
    ) = etl.run_etl(args.movies_csv_data_location, args.ratings_csv_data_location)

    transformed_train_dataset_dir = os.path.join(
        args.etl_output_dir, "transformed_data/train"
    )
    logging.info(
        f"Writting transformed training data to {transformed_train_dataset_dir}"
    )
    transformed_train_dataset.to_parquet(
        output_path=transformed_train_dataset_dir,
        shuffle=nvt.io.Shuffle.PER_PARTITION,
        cats=features.CATEGORICAL_FEATURE_NAMES,
        labels=features.TARGET_FEATURE_NAME,
        dtypes=features.get_dtype_dict(),
    )
    logging.info("Train data parquet files are written.")

    transformed_test_dataset_dir = os.path.join(
        args.etl_output_dir, "transformed_data/test"
    )
    logging.info(
        f"Writting transformed testing data to {transformed_test_dataset_dir}"
    )
    transformed_test_dataset.to_parquet(
        output_path=transformed_test_dataset_dir,
        shuffle=False,
        cats=features.CATEGORICAL_FEATURE_NAMES,
        labels=features.TARGET_FEATURE_NAME,
        dtypes=features.get_dtype_dict(),
    )
    logging.info("Test data parquet files are written.")

    logging.info("Saving transformation workflow...")
    transform_workflow.save(LOCAL_TRANSFORM_DIR)
    logging.info("Transformation workflow is saved.")

    logging.info("Uploading transform workflow to Cloud Storage...")
    utils.upload_directory(
        LOCAL_TRANSFORM_DIR, os.path.join(args.etl_output_dir, "transform_workflow")
    )
    try:
        tf.io.gfile.rmtree(LOCAL_TRANSFORM_DIR)
        tf.io.gfile.rmtree("categories")
    except:
        pass
    logging.info("Transformation uploaded to Cloud Storage.")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Python Version = {sys.version}")
    logging.info(f"TensorFlow Version = {tf.__version__}")
    logging.info(f'TF_CONFIG = {os.environ.get("TF_CONFIG", "Not found")}')
    logging.info(f"DEVICES = {device_lib.list_local_devices()}")
    logging.info(f"Task started...")
    main()
    logging.info(f"Task completed.")
