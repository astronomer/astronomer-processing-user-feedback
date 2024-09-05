# Anyscale + Astronomer for processing user feedback - Reference architecture

Welcome! 
This project is an end-to-end pipeline showing how to process user feedback in production with [Anyscale](https://anyscale.com/) and [Astronomer](https://www.astronomer.io/) 
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
- [Anyscale](https://www.anyscale.com/) to run the fine-tuning jobs and to deploy the LLM as a service
- [Amazon S3](https://aws.amazon.com/s3/) free tier. You can also adapt the pipeline to run with your preferred object storage solution. Or re-use the cloud bucket that you configured on the Anyscale platform

Optional:

- [Snowflake](https://www.snowflake.com/en/). A [free trial](https://signup.snowflake.com/) is available.

## How to set up the demo environment

Follow the steps below to set up the demo for yourself.

1. Install Astronomer's open-source local Airflow development tool, the [Astro CLI](https://www.astronomer.io/docs/astro/cli/overview).
2. Log into your Anyscale account and complete the following steps -
  - Setup up a Cloud provider (AWS/GCP) that the Anyscale Platform can use to run your jobs
  - Configure your Anyscale connection using the [section](#configuration-details-for-anyscale-connection) section below. We will need these details while setting up our Airflow job
3. Configure your AWS connection as shown in the [section](#configuration-details-for-aws-connection) below
4. (Optional): Sign up for a [free trial](https://signup.snowflake.com/) of Snowflake. See the [section](#configuration-details-for-snowflake-connection) below for more information on setting up your snowflake connection

    A Snowflake account is needed to run the following DAGs, which track the number of records we read from each input.
    
    - [`data_update`](dags/automated_retraining/data_update.py)

    If you don't have a Snowflake account, delete the `write_metrics_to_snowflake` task from the dag.

4. Fork this repository and clone the code locally.


### Configuration Details for Anyscale Connection

To integrate Airflow with Anyscale, you will need to provide several configuration details:

- **Anyscale API Token**: Obtain your API token either by using the anyscale cli or through the [Anyscale console](https://console.anyscale.com/v2/api-keys?api-keys-tab=platform).

- **Compute Config (optional)**: If you want to constrain autoscaling, you can specify the compute cluster that will execute your Ray script by either:
  - Dynamically providing this via the `compute_config` input parameter, or
  - Creating a compute configuration in Anyscale and using the resulting ID in the `compute_config_id` parameter.

- **Image URI**: Specify the docker image you would like your operator to use. Make sure your image is accessible within your Anyscale account. Note, you can alternatively specify a containerfile that can be used to dynamically build the image

### Configuration Details for AWS Connection

To integrate Airflow with AWS, we will need to setup an AWS connection. To do so, we need to collect the following information for your AWS user

- `AWS_ACCESS_KEY`
- `AWS_SECRET_ACCESS_KEY`

Alternatively, see the [Authentication and access credentials](https://docs.aws.amazon.com/cli/v1/userguide/cli-chap-authentication.html) page for other ways of authenticating with AWS


### Configuration Details for Snowflake Connection

To integrate Airflow with Snowflake, we will need to setup a Snowflake Connection. To do so, we will need the following details

- Schema
- Login & Password
- Account
- Data warehouse
- Database
- Region

We will also provide the table name within the task.

The demo is setup with a schema called `DEMOUSER`, a database called `SANDBOX` and table called `ASTRONOMER_ANYSCALE_DEMO`. You may also use your own tables in which case, don't forget to update the task code


### Run the project locally

1. Create a new file called .env in the root of the cloned repository and copy the contents of .env_example into it. Fill out the placeholders with your own values for `BUCKET`, `ORG_ID`, `CLOUD_ID`, `HF_TOKEN` & `ANYSCALE_CLI_TOKEN`

2. In the root of the repository, run `astro dev start` to start up the following Docker containers. This is your local development environment.

    - **Postgres**: Airflow's Metadata Database.
    - **Webserver**: The Airflow component responsible for rendering the Airflow UI. Accessible on port `localhost:8080`.
    - **Scheduler**: The Airflow component responsible for monitoring and triggering tasks
    - **Triggerer**: The Airflow component responsible for triggering deferred tasks

    Note that after any changes to .env you will need to run astro dev start for new environment variables to be picked up.

3. Access the Airflow UI at `localhost:8080` and follow the DAG running instructions in the [Running the DAGs](#running-the-dags) section of this README. You can run and develop DAGs in this environment.

### Run the project in the cloud

1. Sign up to [Astro](http://qrco.de/bfHv2Q) for free and follow the onboarding flow to create a deployment with default configurations.
3. Deploy the project to Astro using `astro deploy`. See [Deploy code to Astro](https://www.astronomer.io/docs/astro/deploy-code).
3. Set up your Anyscale, AWS and Snowflake connections, as well as all other environment variables listed in [`.env_example](.env_example) on Astro. For instructions see [Manage Airflow connections and variables](https://www.astronomer.io/docs/astro/manage-connections-variables) and [Manage environment variables on Astro](https://www.astronomer.io/docs/astro/manage-env-vars).


4. Open the Airflow UI of your Astro deployment and follow the steps in [Running the DAGs](#running-the-dags).


## Running the DAGs

This repo contains three DAGs:

1. Access_and_transform_data
2. Finetune_llm_and_deploy_challenger
3. llm_retrain_and_rollout

Unpause all DAGs by toggling the switch to the left of each DAG name.

![Screenshot of the Airflow UI showing the DAGs unpaused.](/static/unpaused.png)

### 1. Access_and_transform_data

This DAG demonstrates data processing from multiple sources, transformation using dynamic task mapping, joining, and uploading to S3.

> [!IMPORTANT]
> **Demo simplifications:**
> - All data is read from S3, despite task names suggesting other sources
> - transform_data task passes data through unchanged (customize as needed)
> - upload_to_s3 task saves to the same S3 folder used for reading

The DAG concludes by uploading a Dataset called `data_transformed`, which triggers the next DAG.

![Screenshot of the data update DAG](/static/DAG-1.png)

### 2. Finetune_llm_and_deploy_challenger

Triggered by the `data_transformed` Dataset update, this DAG reads data from S3 and checks for ≥500 records. If met, it initiates fine-tuning using Anyscale's LLM-Forge tool, which automates the process based on a YAML config.

> [!IMPORTANT]
> **Demo simplification:**
> - Data is read from the same S3 folder as the previous DAG

![Screenshot of the model update DAG](/static/DAG-2.png)

### 3. llm_retrain_and_rollout

This manually triggered DAG performs complex champion-challenger analysis.

> [!TIP]
> This job runs for over 2 hours. Adjust operator timeout settings to ensure completion.

![Screenshot of the model update DAG](/static/DAG-3.png)