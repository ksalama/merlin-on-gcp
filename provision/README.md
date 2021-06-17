# (WIP) Provision Google Cloud environment for running Nvidia Merlin with Vertex AI

You can use the [Terraform](https://www.terraform.io/) scripts in the [terraform](terraform) folder to automatically provision the environment required by the samples. 

The scripts perform the following actions:
1. Enable the required Cloud APIs.
2. Create a regional GCS bucket.
3. Create an instance of Vertex Notebooks.
4. Create service accounts for Vertex Training and Vertex Pipelines.