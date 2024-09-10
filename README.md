# Anyscale + Astronomer for processing user feedback - Reference architecture

Welcome! 

This project is an end-to-end pipeline showing how to process user feedback in production with [Anyscale](https://anyscale.com/), an AI compute platform built on [Ray](https://www.ray.io/), and [Astronomer](https://www.astronomer.io/), the best place to run [Apache Airflow®](https://airflow.apache.org/).

This example project structures video gaming feedback, but you can adapt it to build your own pipelines feeding fine-tuning data about any topic to fine-tune models on Anyscale.

> [!TIP]
> If you are new to Airflow, we recommend checking out our get started resources: [DAG writing for data engineers and data scientists](https://www.astronomer.io/events/webinars/dag-writing-for-data-engineers-and-data-scientists-video/) before diving into this project.

## Use case

This repository contains a demo integration of Apache Airflow® with Anyscale to fine-tune a Large Language Model (LLM) for the purpose of structuring video game feedback. The LLM is then deployed using [Anyscale Services](https://docs.anyscale.com/1.0.0/services/get-started/) to process new user feedback relating to video games and extract structured data from it. 

- **Usecase:** Functional representation of unstructured text i.e. extracting structure from unstructured data. This can be extremely helpful in the context of customer support, where a user conversation needs to be correctly classified and specific metadata extracted from it. 
- **Model**: [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- **Dataset**: 
  - [VIGGO dataset](https://huggingface.co/datasets/GEM/viggo) inverted (i.e. inputs are text and labels are the structured data)
    - Initial curated dataset of 100 samples to fine-tune on
    - Expanded dataset of 500 samples for automatic re-fine-tuning
- **Example**:
    - Input: "What is it about games released in 2005 that makes you think it's such a fantastic year for video games?"
    - Output: `request_explanation(release_year[2005], rating[excellent])`

## Project Structure

- **Development** - See [`notebook/`](notebook/develop.ipynb) for the initial development notebook, which shows how to:
  - Deploy a base model as a service
  - Run evaluations against a base model to establish baseline
  - Fine-tune an LLM
  - Deploy the fine-tuned LLM
  - Run an evaluation and compare it to the baseline

- **Automated retraining** - See [`dags/automated_retraining/`](dags/automated_retraining/) for the automated retraining DAGs:

  - DAG 1: [`dags/automated_retraining/data_update.py`](dags/automated_retraining/data_update.py) - regularly updates the fine-tuning data, it:
    - Connects to several data sources (to simplify the demo, the sources are mocked using VIGGO dataset data located in S3)
    - Pre-processes and joins the data

  - DAG 2: [`dags/automated_retraining/model_update.py`](dags/automated_retraining/model_update.py) - shows basic model fine-tuning and deployment, it:
    - Fetches the number of records stored in S3.
      - If > 200 records are found, the pipeline proceeds
      - If less than 200 records are found, the pipeline ends 
    - Fine-tunes an LLM using an [Anyscale Job](https://docs.anyscale.com/1.0.0/jobs/get-started/)
    - Deploys the fine-tuned LLM using a [Anyscale Service](https://docs.anyscale.com/1.0.0/services/get-started/)
  
  - DAG 3: [`dags/automated_retraining/retrain.py`](dags/automated_retraining/retrain.py) - shows advanced on-demand fine-tuning in a champion-challenger pattern, it:
    - Fine-tunes a challenger model using an Anyscale Job
    - Performs a silent rollout of the fine-tuned challenger model using a Anyscale Service 
    - Compares the challenger model to the current champion model
    - If the performance of challenger model is better:
      - Completes rollout of the challenger model
    - If the challenger model is not better:
      - Sends an alert (e.g. via slack) to AI team and does not complete the rollout of the challenger model
    - Sends a fine-tuning report

## Tools Used

- [Apache Airflow®](https://airflow.apache.org/docs/apache-airflow/stable/index.html) running on [Astro](https://www.astronomer.io/product/). A [free trial](http://qrco.de/bfHv2Q) is available.
- [Anyscale](https://www.anyscale.com/), an AI compute platform built on [Ray](https://www.ray.io/), to run the fine-tuning jobs and to deploy the LLM as a service.
- [Amazon S3](https://aws.amazon.com/s3/) free tier. You can also adapt the pipeline to run with your preferred object storage solution or re-use the cloud bucket that you configured on the Anyscale platform.

Optional:

- [Snowflake](https://www.snowflake.com/en/) to store metrics. A [free trial](https://signup.snowflake.com/) is available.

## How to set up the demo environment

Follow the steps below to set up the demo for yourself.

1. Install Astronomer's open-source local Airflow development tool, the [Astro CLI](https://www.astronomer.io/docs/astro/cli/overview).
2. Log into your Anyscale account and complete the following steps:

  - Setup up a Cloud provider (for example [AWS](https://docs.anyscale.com/1.0.0/cloud-deployment/aws/manage-clouds/) or [GCP](https://docs.anyscale.com/1.0.0/cloud-deployment/aws/manage-clouds/)) that the Anyscale Platform can use to run your jobs.
  - Configure your Anyscale connection using the [Configuration Details for your Anyscale Connection](#configuration-details-for-your-anyscale-connection) section below. We will need these details while setting up our Airflow job!

3. Configure your AWS connection as shown in the [Configuration Details for your AWS Connection](#configuration-details-for-your-aws-connection) section below.
4. PLACEHOLDER: upload the VIGGO dataset with the right keys to the S3 bucket - maybe provide the 2 files in the repo?
5. (Optional): Sign up for a [free trial](https://signup.snowflake.com/) of Snowflake. See the [Configuration Details for your Snowflake Connection](#configuration-details-for-your-snowflake-connection) below for more information on setting up your Snowflake connection.

    A Snowflake account is needed to run one task in the following DAG, which tracks the number of records we read from each input.
    
    - [`data_update`](dags/automated_retraining/data_update.py)

    If you don't have a Snowflake account, delete the `write_metrics_to_snowflake` task from the dag to avoid a DAG Import Error.

6. Fork this repository and clone the code locally.

### Configuration Details for your Anyscale Connection

To integrate Airflow with Anyscale, you will need to provide several configuration details:

- **Anyscale API Token**: Obtain your API token either by using the anyscale cli or through the [Anyscale console](https://console.anyscale.com/v2/api-keys?api-keys-tab=platform). This information will be used while setting the Airflow connection.

- **Org ID**: Obtain the Org ID from your Anyscale Web UI. This will be used to identify which anyscale organization to use while running your jobs or deploying your services. This information will be use in an environment variable below.

- **Cloud ID**: Obtain the unique identifier for your cloud configuration on Anyscale. This identifier is available on the UI and will be used in an environment variable below.

The following two configurations are provided to the operators in the DAG:

- **Compute Config (optional)**: If you want to constrain autoscaling, you can specify the compute cluster that will execute your Ray script by either:
  - Dynamically providing this via the `compute_config` input parameter, or
  - Creating a compute configuration in Anyscale and using the resulting ID in the `compute_config_id` parameter.

- **Image URI**: Specify the docker image you would like your operator to use. Make sure your image is accessible within your Anyscale account. Note, you can alternatively specify a containerfile that can be used to dynamically build the image.

### Configuration Details for your AWS Connection

To integrate Airflow with AWS, we will need to setup an AWS connection. To do so, we need to collect the following information for your AWS user:

- `AWS_ACCESS_KEY`
- `AWS_SECRET_ACCESS_KEY`

Alternatively, see the [Authentication and access credentials](https://docs.aws.amazon.com/cli/v1/userguide/cli-chap-authentication.html) page and the [AWS Airflow provider documentation](https://airflow.apache.org/docs/apache-airflow-providers-amazon/stable/connections/aws.html) for other ways of authenticating Airflow to AWS.

### Configuration Details for your Snowflake Connection

To integrate Airflow with Snowflake, we will need to setup a Snowflake Connection. To do so, we will need the following details:

- Schema
- Login & Password
- Account
- Data warehouse
- Database
- Region

We will also provide the table name within the task.

The demo is setup with a schema called `DEMOUSER`, a database called `SANDBOX` and table called `ASTRONOMER_ANYSCALE_DEMO`. You may also use your own tables in which case, don't forget to update the table name in the Airflow task code.

Note that it is also possible to connect to Snowflake using a public-private key pair. See the [Snowflake Airflow provider documentation](https://airflow.apache.org/docs/apache-airflow-providers-snowflake/stable/connections/snowflake.html) for more information.

### Run the project locally

1. Create a new file called `.env` in the root of the cloned repository and copy the contents of `.env_example` into it. Fill out the placeholders with your own values 
    
    - `BUCKET`: The S3 bucket that should be used by Anyscale
    - `ORG_ID`: The Organization to use while running your jobs or deploying your services
    - `CLOUD_ID`: The Identifier for the cloud configuration specified in your Anyscale UI
    - `HF_TOKEN`: The base model is downloaded from Hugging Face and we will therefore all need a HF access token

2. To run the `data_update` DAG, upload the files in the `data/` folder to your AWS S3 bucket and update the keys in `read_discord_data` and `read_twitch_data` tasks.

3. In the root of the repository, run `astro dev start` to start up the following Docker containers. This is your local development environment.

    - **Postgres**: Airflow's Metadata Database.
    - **Webserver**: The Airflow component responsible for rendering the Airflow UI. Accessible on port `localhost:8080`.
    - **Scheduler**: The Airflow component responsible for monitoring and triggering tasks
    - **Triggerer**: The Airflow component responsible for triggering deferred tasks

    Note that after any changes to .env you will need to run `astro dev start` for new environment variables to be picked up.

4. Access the Airflow UI at `localhost:8080` (log in with `admin` as the username and `admin` as the password) and follow the DAG running instructions in the [Running the DAGs](#running-the-dags) section of this README. You can run and develop DAGs in this local Airflow environment.


### Run the project in the cloud

1. Sign up to [Astro](http://qrco.de/bfHv2Q) for free and follow the onboarding flow to create a deployment with default configurations.
3. Deploy the project to Astro using `astro deploy`. See [Deploy code to Astro](https://www.astronomer.io/docs/astro/deploy-code).
3. Set up your Anyscale, AWS and Snowflake connections, as well as all other environment variables listed in [`.env_example](.env_example) on Astro. For instructions see [Manage Airflow connections and variables](https://www.astronomer.io/docs/astro/manage-connections-variables) and [Manage environment variables on Astro](https://www.astronomer.io/docs/astro/manage-env-vars).
4. Open the Airflow UI of your Astro deployment and follow the steps in [Running the DAGs](#running-the-dags).

## Running the DAGs

This repo contains three DAGs:

1. [`Access_and_transform_data`](dags/automated_retraining/data_update.py)
2. [`Finetune_llm_and_deploy_challenger`](dags/automated_retraining/model_update.py)
3. [`llm_retrain_and_rollout`](dags/automated_retraining/retrain.py)

Unpause all DAGs by toggling the switch to the left of each DAG name.

![Screenshot of the Airflow UI showing the DAGs unpaused.](/static/unpaused.png)

### 1. Access_and_transform_data

This DAG demonstrates data processing from multiple sources, transformation using dynamic task mapping, joining the data, and uploading it to S3. This DAG needs to run before the other two DAGs!

> [!IMPORTANT]
> **Demo simplifications:**
> - All data is read from S3, despite task names suggesting other sources. Please upload data from the data folder to your s3 bucket
> - transform_data task passes data through unchanged (customize as needed)
> - upload_to_s3 task saves to the same S3 folder used for reading

The DAG concludes by updating a [Airflow Dataset](https://www.astronomer.io/docs/learn/airflow-datasets) called `data_transformed`, which triggers the next DAG.

![Screenshot of the data update DAG](/static/DAG-1.png)

### 2. Finetune_llm_and_deploy_challenger

Triggered by the `data_transformed` Dataset update, this DAG reads data from S3 and checks for ≥200 records. If met, it initiates fine-tuning using Anyscale's LLM-Forge tool, which automates the process based on a YAML config.

> [!IMPORTANT]
> **Demo simplification:**
> - Data is read from the same S3 folder as the previous DAG

![Screenshot of the model update DAG](/static/DAG-2.png)

### 3. llm_retrain_and_rollout

This manually triggered DAG performs complex champion-challenger analysis. Its an advanced example of how to use the operators in the Anyscale provider and Anyscale's LLM-Forge tool to fine-tune a model and deploy it

> [!TIP]
> This job runs for over 2 hours. Adjust operator timeout settings to ensure completion.

![Screenshot of the model update DAG](/static/DAG-3.png)