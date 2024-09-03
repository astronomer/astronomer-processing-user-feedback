# Astronomer Anyscale Dynamic Cluster Provisioning

Welcome! 
This project is an end-to-end pipeline showing how to processing user feedback in production with [Anyscale](https://anyscale.com/) and [Astronomer](https://www.astronomer.io/) 
for an eCommerce use case. You can use this project as a starting point to build your own pipelines for similar use cases.

> [!TIP]
> If you are new to Airflow, we recommend checking out our get started resources: [DAG writing for data engineers and data scientists](https://www.astronomer.io/events/webinars/dag-writing-for-data-engineers-and-data-scientists-video/) before diving into this project.

## Use case
A demo integration of Airflow with Anyscale using LLM finetuning as a usecase 

- **Usecase:** Functional representation of unstructured text i.e. extracting structure from unstructured data. This can be extremely helpful in the context of customer support, where a user conversation needs to be correctly classified and specific metadata extracted from it. 
- **Model**: mistralai/Mistral-7B-Instruct-v0.1
- **Dataset**: 
  - VIGGO dataset inverted (i.e. inputs are text and labels are the structured data)
    - Initial curated dataset of 100 samples to finetune on
    - Expanded dataset of 500 samples to automatically finetune on again.
- **Example**:
    - Input: "What is it about games released in 2005 that makes you think it's such a fantastic year for video games?"
    - Output: `request_explanation(release_year[2005], rating[excellent])`

## Project Structure

- **Development** - See `notebook/` for the development notebooks
  - Develop in workspace (just Python code)
  - Deploy a base model as a service
  - Run evaluation against a base model to establish baseline
  - Finetune an LLM
  - Deploy finetune
  - Run evaluation and compare to baseline
- **Automated retraining** - See `dags/automated_retraining/` for the automated retraining DAGs
  - DAG 1: regular data update - See `dags/automated_retraining/data_update.py`
    - Connect to two/three common datasources (Zendesk, slack, ...)
    - Join data
    - Count data
    - If data is large enough:
      - Trigger next DAG (can happen using airflow trigger or airflow SDK)
  - DAG 2: regular model update - See `dags/automated_retraining/model_update.py`
    - Finetune an LLM using an Anysale Job
    - Deploy the LLM using an Anyscale Service
  
  - DAG 3: on-demand finetuning, evaluation and release - See `dags/automated_retraining/retrain.py`
    - Finetune challenger using an Anyscale Job
    - Silent rollout of challenger using an Anyscale Service 
    - Evaluate challenger
    - Send fine tuning report somewhere
    - If performance of challenger is better:
      - Complete rollout of challenger
    - If not:
      - send an alert (e.g. via slack) to AI team

## Tools Used

- [Apache Airflow®](https://airflow.apache.org/docs/apache-airflow/stable/index.html) running on [Astro](https://www.astronomer.io/product/). A [free trial](http://qrco.de/bfHv2Q) is available.
- [Anyscale](https://www.anyscale.com/) to run LLM jobs
- [Amazon S3](https://aws.amazon.com/s3/) free tier. You can also adapt the pipeline to run with your preferred object storage solution.

Optional:

- [Snowflake](https://www.snowflake.com/en/). A [free trial](https://signup.snowflake.com/) is available.

## How to set up the demo environment

Follow the steps below to set up the demo for yourself.

1. Install Astronomer's open-source local Airflow development tool, the [Astro CLI](https://www.astronomer.io/docs/astro/cli/overview).
2. Log into your AWS account and create [a new empty S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/creating-bucket.html). Make sure you have a set of [AWS credentials](https://docs.aws.amazon.com/iam/) with `AmazonS3FullAccess` for this new bucket.
3. (Optional): Sign up for a [free trial](https://signup.snowflake.com/) of Snowflake. Create a database called `ASTRONOMER_ANYSCALE_DEMO` with a schema called `dev`.

    A Snowflake account is needed to run the following DAGs, which add additional product information about sneakers and run the bonus batch inference pipeline.
    
    - [`data_update`](dags/automated_retraining/model_update.py)

    If you don't have a Snowflake account, delete these DAGs by deleting their file in the `dags` folder.

4. Fork this repository and clone the code locally.


### Configuration Details for Anyscale Connection

To integrate Airflow with Anyscale, you will need to provide several configuration details:

- **Anyscale API Token**: Obtain your API token either by using the anyscale cli or through the [Anyscale console](https://console.anyscale.com/v2/api-keys?api-keys-tab=platform).

- **Compute Config (optional)**: If you want to constrain autoscaling, you can specify the compute cluster that will execute your Ray script by either:
  - Dynamically providing this via the `compute_config` input parameter, or
  - Creating a compute configuration in Anyscale and using the resulting ID in the `compute_config_id` parameter.

- **Image URI**: Specify the docker image you would like your operator to use. Make sure your image is accessible within your Anyscale account. Note, you can alternatively specify a containerfile that can be used to dynamically build the image

### Configuration Details for AWS Connection

### Configuration Details for Snowflake Connection


### Run the project locally


### Run the project in the cloud

## Running the DAGs