# Anyscale and Airflow

## astronomer-anyscale-dynamic-cluster-provisioning
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
- **Migration to prod** - See `dags/simple/` for a simple airflow DAG to migrate to prod
  - Finetune an LLM using an Anysale Job
  - Deploy the LLM using an Anyscale Service
  - Run an evaluation as an Anyscale Job
- **Automated retraining** - See `dags/automated_retraining/` for the automated retraining DAGs
  - DAG 1: regular data update - See `dags/automated_retraining/data_update.py`
    - Connect to two/three common datasources (Zendesk, slack, ...)
    - Join data
    - Count data
    - If data is large enough:
      - Trigger next DAG (can happen using airflow trigger or airflow SDK)
  - DAG 2: on-demand finetuning, evaluation and release - See `dags/automated_retraining/retrain.py`
    - Finetune challenger using an Anyscale Job
    - Silent rollout of challenger using an Anyscale Service 
    - Evaluate challenger
    - Send fine tuning report somewhere
    - If performance of challenger is better:
      - Complete rollout of challenger
    - If not:
      - send an alert (e.g. via slack) to AI team
