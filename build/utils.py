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
"""Utilities for deploying pipelines and models to Vertex AI."""


import argparse
import os
import sys
import logging
import json

from google.cloud import aiplatform as vertex_ai


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode', 
        type=str,
    )

    parser.add_argument(
        '--project',  
        type=str,
    )
    
    parser.add_argument(
        '--region',  
        type=str,
    )
    
    parser.add_argument(
        '--endpoint-display-name', 
        type=str,
    )

    parser.add_argument(
        '--model-display-name', 
        type=str,
    )
    
    parser.add_argument(
        '--pipeline-name', 
        type=str,
    )

    return parser.parse_args()


def compile_pipeline(pipeline_name):
    from src.kfp_pipelines import runner
    pipeline_definition_file = f"{pipeline_name}.json"
    pipeline_definition = runner.compile_pipeline(pipeline_definition_file)
    return pipeline_definition

    

def main():
    args = get_args()
        
    if args.mode == 'compile-pipeline':
        if not args.pipeline_name:
            raise ValueError("pipeline-name must be supplied.")
            
        result = compile_pipeline(args.pipeline_name)

    else:
        raise ValueError(f"Invalid mode {args.mode}.")
        
    logging.info(result)
        
    
if __name__ == "__main__":
    main()
    