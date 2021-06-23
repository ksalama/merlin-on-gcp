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
"""Common utilities."""

import os
import tensorflow.io as tf_io


def upload_directory(source_dir, destination_dir):
    
    stack = [source_dir]
    while stack:
        current = stack.pop()
        if tf_io.gfile.isdir(current):
            for next_item in tf_io.gfile.listdir(current):
                stack.append(os.path.join(current, next_item))
        else:
            x = current.replace(source_dir + "/", '')
            dist = os.path.join(destination_dir, x)
            tf_io.gfile.copy(
                current, 
                dist,
                overwrite=True
                
            )
            
            
def download_directory(source_dir, destination_dir):
    
    source_stack = [source_dir]
    destination_stack = [destination_dir]
    
    while source_stack:

        source = source_stack.pop()
        if source[-1] == '/':
            source = source[:-1]
        
        destination = destination_stack.pop()
        base_name = os.path.basename(source)
        destination = os.path.join(destination, base_name)

        if tf_io.gfile.isdir(source):
            tf_io.gfile.mkdir(destination)
            for next_item in tf_io.gfile.listdir(source):
                destination_stack.append(destination)
                source_stack.append(os.path.join(source, next_item))
        else:
            tf_io.gfile.copy(
                source, 
                destination,
                overwrite=True
            )
            
            
            
def copy_files(file_pattern, destination_dir):
        
    for file_path in tf_io.gfile.glob(file_pattern):
        file_name = os.path.basename(file_path)
        tf_io.gfile.copy(file_path, os.path.join(destination_dir, file_name), overwrite=True)
