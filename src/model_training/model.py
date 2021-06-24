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
"""A DNN Keras regression model."""


import nvtabular as nvt
import tensorflow as tf
from nvtabular.loader.tensorflow import KerasSequenceLoader, KerasSequenceValidater
from nvtabular.framework_utils.tensorflow import layers

from src.common import features


def create_inputs():

    inputs = {}
    for feature_name in features.CATEGORICAL_FEATURE_NAMES:
        inputs[feature_name] = tf.keras.Input(
            name=feature_name, dtype=tf.int32, shape=(1,)
        )

    for feature_name in features.MULTIVALUE_FEATURE_NAMES:
        inputs[feature_name] = (
            tf.keras.Input(name=f"{feature_name}__values", dtype=tf.int64, shape=(1,)),
            tf.keras.Input(name=f"{feature_name}__nnzs", dtype=tf.int64, shape=(1,)),
        )
    return inputs


def create_embedding_layers(embedding_shapes):

    embedding_layers = []
    for feature_name in features.get_categorical_feature_names():
        embedding_layers.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_identity(
                    feature_name, embedding_shapes[feature_name][0]
                ),  # Embedding input dimension
                embedding_shapes[feature_name][1],  # Embedding output dimension
            )
        )
    return embedding_layers


def create(embedding_shapes, hidden_units):

    inputs = create_inputs()
    embedding_layers = create_embedding_layers(embedding_shapes)

    x = layers.DenseFeatures(embedding_layers)(inputs)
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)

    logits = tf.keras.layers.Dense(1, activation="sigmoid", name="logits")(x)
    model = tf.keras.Model(inputs=inputs, outputs=logits)

    return model
