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
"""Data preprocessing using NVTabular."""

import os
import logging

import tensorflow.io as tf_io
import cudf
import nvtabular as nvt

from google.cloud import aiplatform as vertex_ai

from sklearn.model_selection import train_test_split
from src.common import features, utils


LOCAL_TRANSFORM_DIR = 'transform_workflow'
RANDOM_STATE = 42


def get_dataset_gcs_location(dataset_display_name):
    datasets = vertex_ai.TabularDataset.list()
    dataset = None
    for entry in datasets:
        if entry.display_name == dataset_display_name:
            dataset = entry
            break

    if not dataset:
        raise ValueError(f"Dataset with display name {dataset_display_name} does not exist!")
    
    return dataset.gca_resource.metadata['inputConfig']['gcsSource']['uri'][0]



def prep_dataframe(dataframe):
    
    for feature_name in features.UNUSED_FEATURES:
        if feature_name in list(dataframe.columns):
            dataframe.drop(feature_name, axis=1, inplace=True)
    
    for feature_name in features.MULTIVALUE_FEATURE_NAMES:
         if feature_name in list(dataframe.columns):
                dataframe[feature_name] = dataframe[feature_name].str.split("|")
                
    return dataframe


def create_workflow(movies_df):
    joined = ["userId", "movieId"] >> nvt.ops.JoinExternal(movies_df, on=["movieId"])
    cat_features = joined >> nvt.ops.Categorify()
    ratings = nvt.ColumnGroup(["rating"]) >> (lambda col: (col > 3).astype("int8"))
    output = cat_features + ratings
    workflow = nvt.Workflow(output)
    return workflow



def run_etl(
    project, 
    region, 
    movies_dataset_display_name, 
    ratings_dataset_display_name, 
    etl_output_dir,
    test_size=0.2):
    
    vertex_ai.init(
        project=project,
        location=region)
    
    logging.info("Getting GCS data locations...")
    movies_csv_data_location = get_dataset_gcs_location(movies_dataset_display_name)
    ratings_csv_data_location = get_dataset_gcs_location(ratings_dataset_display_name)
    
    logging.info("Loading dataframes...")
    movies_dataframe = prep_dataframe(cudf.read_csv(movies_csv_data_location))
    ratings_dataframe = prep_dataframe(cudf.read_csv(ratings_csv_data_location))
    logging.info("Dataframe loaded.")
    
    logging.info(f"Movies data: {movies_dataframe.shape}.")
    logging.info(f"Ratings data: {ratings_dataframe.shape}.")
    
    logging.info("Splitting dataset to train and test splits...")
    train_split, test_split = train_test_split(
        ratings_dataframe, test_size=test_size, random_state=RANDOM_STATE)
    logging.info(f"Train split size: {len(train_split.index)}")
    logging.info(f"Test split size: {len(test_split.index)}")

    logging.info("Loading NVTabular datasets...")
    train_dataset = nvt.Dataset(train_split)
    test_dataset = nvt.Dataset(test_split)
    logging.info("NVTabular datasets loaded.")
    
    logging.info("Creating transformation workflow...")
    workflow = create_workflow(movies_dataframe)
    logging.info("Fitting workflow to train data split...")
    workflow.fit(train_dataset)
    logging.info("Transformation workflow is fitted.")
    
    logging.info("Transforming train dataset...")
    workflow.transform(train_dataset).to_parquet(
        output_path=os.path.join(etl_output_dir, "transformed_data/train"),
        shuffle=nvt.io.Shuffle.PER_PARTITION,
        cats=features.CATEGORICAL_FEATURE_NAMES,
        labels=features.TARGET_FEATURE_NAME,
        dtypes=features.get_dtype_dict(),
    )
    logging.info("Train data is transformed.")
    
    logging.info(f"Transforming test dataset...")
    workflow.transform(test_dataset).to_parquet(
        output_path=os.path.join(etl_output_dir, "transformed_data/test"),
        shuffle=False,
        cats=features.CATEGORICAL_FEATURE_NAMES,
        labels=features.TARGET_FEATURE_NAME,
        dtypes=features.get_dtype_dict(),
    )
    logging.info("Test dataset is transformed.")
    
    logging.info("Saving transformation workflow...")
    workflow.save(LOCAL_TRANSFORM_DIR)
    logging.info("Transformation workflow is saved.")
    
    logging.info("Uploading trandorm workflow to Cloud Storage...")
    utils.upload_directory(LOCAL_TRANSFORM_DIR, etl_output_dir)
    try:
        tf_io.gfile.rmtree(LOCAL_TRANSFORM_DIR)
        tf_io.gfile.rmtree("categories")
    except: pass
    logging.info("Transformation uploaded to Cloud Storage.")
    


    

    
