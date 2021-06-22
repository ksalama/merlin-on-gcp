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
"""Train and evaluate the model."""

import os

os.environ["TF_MEMORY_ALLOCATION"] = "0.7"  # fraction of free memory
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import logging
import tensorflow as tf
from tensorflow import keras
import nvtabular as nvt
from nvtabular.loader.tensorflow import KerasSequenceLoader, KerasSequenceValidater
#from nvtabular.inference.triton import export_tensorflow_ensemble

from src.common import features, utils
from src.model_training import model

HIDDEN_UNITS = [128, 128]
LEARNING_RATE = 0.001
BATCH_SIZE = 512
NUM_EPOCHS = 1


def update_hyperparams(hyperparams: dict) -> dict:
    if "hidden_units" not in hyperparams:
        hyperparams["hidden_units"] = HIDDEN_UNITS
    else:
        if not isinstance(hyperparams["hidden_units"], list):
            hyperparams["hidden_units"] = [
                int(v) for v in hyperparams["hidden_units"].split(",")
            ]
    if "learning_rate" not in hyperparams:
        hyperparams["learning_rate"] = LEARNING_RATE
    if "batch_size" not in hyperparams:
        hyperparams["batch_size"] = BATCH_SIZE
    if "num_epochs" not in hyperparams:
        hyperparams["num_epochs"] = NUM_EPOCHS
    return hyperparams



def train(
    train_data_file_pattern,
    nvt_workflow,
    hyperparams,
    log_dir):
    
    hyperparams = update_hyperparams(hyperparams)
    logging.info("Hyperparameter:")
    logging.info(hyperparams)
    logging.info("")
    
    logging.info("Preparing train dataset loader...")
    train_dataset = KerasSequenceLoader(
        train_data_file_pattern,
        batch_size=hyperparams['batch_size'],
        label_names=features.TARGET_FEATURE_NAME,
        cat_names=features.get_categorical_feature_names(),
        cont_names=features.NUMERICAL_FEATURE_NAMES,
        engine="parquet",
        shuffle=True,
        buffer_size=0.06,  # how many batches to load at once
        parts_per_chunk=1,
    )
    
    embedding_shapes, embedding_shapes_multihot = nvt.ops.get_embedding_sizes(nvt_workflow)
    embedding_shapes.update(embedding_shapes_multihot)
    logging.info(f"Embedding shapes: {embedding_shapes}")
    
    hidden_units = hyperparams['hidden_units']
    
    recommendation_model = model.create(embedding_shapes, hidden_units)
    
    optimizer = keras.optimizers.Adam(learning_rate=hyperparams["learning_rate"])
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [keras.metrics.MeanAbsoluteError(name="mae")]
    
    logging.info("Compiling the model...")
    recommendation_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    logging.info("Model fitting started...")
    history = recommendation_model.fit(
        train_dataset,
        epochs=hyperparams['num_epochs'],
        #callbacks=[evaluation_callback], 
    )
    logging.info("Model fitting finished.")
    
    return recommendation_model



def evaluate(
    recommendation_model,
    eval_data_file_pattern,
    hyperparams):
    
    logging.info("Preparing evaluation dataset loader...")
    eval_dataset = KerasSequenceLoader(
        eval_data_file_pattern,
        batch_size=hyperparams['batch_size'],
        label_names=features.TARGET_FEATURE_NAME,
        cat_names=features.get_categorical_feature_names(),
        cont_names=features.NUMERICAL_FEATURE_NAMES,
        engine="parquet",
        shuffle=False,
        buffer_size=0.06,  # how many batches to load at once
        parts_per_chunk=1,
    )
    
    logging.info("Evaluating the model...")
    evaluation_metrics =  recommendation_model.evaluate(eval_dataset)
    logging.info(f"Evaluation loss: {evaluation_metrics[0]} - Evaluation MAE {evaluation_metrics[1]}")
    return evaluation_metrics



# def export(recommendation_model, nvt_workflow, model_name, export_dir):
    
#     for feature_name in features.CATEGORICAL_FEATURE_NAMES:
#         nvt_workflow.output_dtypes[feature_name] = "int32"

#     export_tensorflow_ensemble(
#         recommendation_model, 
#         nvt_workflow, 
#         model_name, 
#         export_dir, features.TARGET_FEATURE_NAME
#     )
    




