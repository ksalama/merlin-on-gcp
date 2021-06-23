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
import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse

from src.data_preprocessing import etl


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
        type=str,
    )
    
    parser.add_argument(
        "--ratings-csv-data-location",
        type=str,
    )

    parser.add_argument(
        "--ratings-dataset-display-name",
        type=str,
    )

    parser.add_argument(
        "--etl-output-dir",
        type=str,
    )

    parser.add_argument(
        "--test-size", 
        default=0.2, type=float
    )

    parser.add_argument(
        "--project", 
        type=str
    )
    
    parser.add_argument(
        "--region", 
        type=str
    )

    return parser.parse_args()


def main():
    args = get_args()

    etl.run_etl(
        args.project, 
        args.region, 
        args.movies_csv_data_location, 
        args.ratings_csv_data_location, 
        args.etl_output_dir
    )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Python Version = {sys.version}")
    logging.info(f"TensorFlow Version = {tf.__version__}")
    logging.info(f'TF_CONFIG = {os.environ.get("TF_CONFIG", "Not found")}')
    logging.info(f"DEVICES = {device_lib.list_local_devices()}")
    logging.info(f"Task started...")
    main()
    logging.info(f"Task completed.")