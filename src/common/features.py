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
"""Features metadata utils."""

import numpy as np


MOVIES_CSV_COLUMNS = [
    "movieId",
    "title",
    "genres"

]

RATINGS_CSV_COLUMNS = [
    "userId",
    "movieId",
    "rating", 
    "timestamp"
]

UNUSED_FEATURES = [
    "timestamp",
    "title"
]

MULTIVALUE_FEATURE_NAMES = [
    "genres"
]

CATEGORICAL_FEATURE_NAMES = ["userId", "movieId"]

TARGET_FEATURE_NAME = ["rating"]


def get_dtype_dict():
    
    dtypes_dict = {}

    for feature_name in CATEGORICAL_FEATURE_NAMES:
        dtypes_dict[feature_name] = np.int64

    dtypes_dict[TARGET_FEATURE_NAME[0]] = np.float32
    
    return dtypes_dict