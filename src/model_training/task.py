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
"""The entrypoint for the Vertex AI training job."""


import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]='python'
os.environ["TF_MEMORY_ALLOCATION"]='0.7'

import sys
from datetime import datetime
import logging
import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse

import nvtabular as nvt

# from google.cloud import aiplatform as vertex_ai
from google.protobuf.internal import api_implementation

from src.model_training import trainer
from src.common import utils

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        default=os.getenv("AIP_MODEL_DIR"),
        type=str,
    )

    parser.add_argument(
        "--log-dir",
        default=os.getenv("AIP_TENSORBOARD_LOG_DIR"),
        type=str,
    )

    parser.add_argument(
        "--train-data-file-pattern",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--test-data-file-pattern",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--transform_workflow_dir",
        required=True,
        type=str,
    )

    parser.add_argument("--learning-rate", default=0.001, type=float)

    parser.add_argument("--batch-size", default=2048, type=float)

    parser.add_argument("--hidden-units", default="128,128", type=str)

    parser.add_argument("--num-epochs", default=1, type=int)

    #     parser.add_argument("--project", type=str)
    #     parser.add_argument("--region", type=str)
    #     parser.add_argument("--staging-bucket", type=str)
    #     parser.add_argument("--experiment-name", type=str)
    #     parser.add_argument("--run-name", type=str)

    return parser.parse_args()


def main():
    args = get_args()

    experiment_params = vars(args)
    experiment_params = trainer.update_hyperparams(experiment_params)
    logging.info(f"Parameter values: {experiment_params}")

    #     if args.experiment_name:
    #         vertex_ai.init(
    #             project=args.project,
    #             location=args.region,
    #             staging_bucket=args.staging_bucket,
    #             experiment=args.experiment_name,
    #         )
    #         logging.info(f"Using Vertex AI experiment: {args.experiment_name}")

    #         run_id = args.run_name
    #         if not run_id:
    #             run_id = f"run-gcp-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    #         vertex_ai.start_run(run_id)
    #         logging.info(f"Run {run_id} started.")

    #     if args.experiment_name:
    #         vertex_ai.log_params(experiment_params)

    logging.info("Downloading data and transform workflow...")
    if tf.io.gfile.exists("data"):
        tf.io.gfile.rmtree("data")
    if tf.io.gfile.exists("transform_workflow"):
        tf.io.gfile.rmtree("transform_workflow")

    tf.io.gfile.mkdir("data")
    tf.io.gfile.mkdir("data/train")
    tf.io.gfile.mkdir("data/test")

    utils.copy_files(args.train_data_file_pattern, "data/train")
    utils.copy_files(args.test_data_file_pattern, "data/test")
    utils.download_directory(args.transform_workflow_dir, ".")
    logging.info("Data and workflow are downloaded.")

    logging.info(f"Loading nvt workflow...")
    nvt_workflow = nvt.Workflow.load("transform_workflow")
    logging.info(f"nvt workflow loaded.")

    recommendation_model = trainer.train(
        train_data_file_pattern="data/train/*.parquet",
        nvt_workflow=nvt_workflow,
        hyperparams=experiment_params,
        log_dir=args.log_dir,
    )

    val_loss, val_mae = trainer.evaluate(
        recommendation_model,
        eval_data_file_pattern="data/test/*.parquet",
        hyperparams=experiment_params,
    )

#    if args.experiment_name:
#        vertex_ai.log_metrics({"val_loss": val_loss, "val_accuracy": val_accuracy})

    trainer.export(recommendation_model, nvt_workflow, model_name, args.model_dir)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Proto API implementation: {api_implementation.Type()}")
    logging.info(f"Python Version = {sys.version}")
    logging.info(f"TensorFlow Version = {tf.__version__}")
    logging.info(f'TF_CONFIG = {os.environ.get("TF_CONFIG", "Not found")}')
    logging.info(f"DEVICES = {tf.config.list_physical_devices()}")
    logging.info(f"Task started...")
    main()
    logging.info(f"Task completed.")
