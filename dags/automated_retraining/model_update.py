"""Simple DAG for LLM finetuning and evaluation."""

import os
import pandas as pd
import io
from pathlib import Path
from typing import Optional
from pendulum import datetime
from anyscale_provider.operators.anyscale import (
    SubmitAnyscaleJob,
    RolloutAnyscaleService,
)
from anyscale_provider.hooks.anyscale import AnyscaleHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.datasets import Dataset

DAGS_DIR = Path(__file__).parent.parent

def count_records_in_s3_file(bucket: str, key: str, **kwargs) -> int:
    s3_hook = S3Hook(aws_conn_id='aws_conn')
    data = s3_hook.read_key(key=key, bucket_name=bucket)
    df = pd.read_json(io.StringIO(data), lines=True)
    record_count = len(df)
    
    # Store the record count in XCom
    ti = kwargs['ti']
    ti.xcom_push(key='record_count', value=record_count)
    
    return record_count

def check_record_count(**kwargs):
    ti = kwargs['ti']
    record_count = ti.xcom_pull(key='record_count', task_ids='record_count')
    if record_count >= 200:
        return 'get_lora_storage_path'
    else:
        return 'end_dag'

def get_lora_storage_path(version: int) -> str:
    """Get the path to the LORA storage."""
    bucket = os.environ["BUCKET"]
    org_id = os.environ["ORG_ID"]
    cloud_id = os.environ["CLOUD_ID"]

    return f"s3://{bucket}/{org_id}/{cloud_id}/artifact_storage/lora_fine_tuning/v{version}/"


def build_service_applications_spec(
    base_model: str, model_storage_path: str
) -> list[dict]:
    deploy_config_path = DAGS_DIR / "configs" / "deploy" / base_model / "config.yaml"
    model_relative_config_path = str(deploy_config_path.relative_to(DAGS_DIR))

    return [
        {
            "name": "llm-endpoint",
            "route_prefix": "/",
            "import_path": "aviary_private_endpoints.backend.server.run:router_application",
            "args": {
                "models": [],
                "multiplex_models": [model_relative_config_path],
                "dynamic_lora_loading_path": model_storage_path,
            },
            "runtime_env": {
                "env_vars": {"HUGGING_FACE_HUB_TOKEN": os.environ["HF_TOKEN"]}
            },
        }
    ]


def get_service_url_and_token(service_name: str) -> tuple[str, str]:
    anyscale = AnyscaleHook(conn_id="anyscale_conn").client
    status = anyscale.service.status(name=service_name)
    return status.query_url, status.query_auth_token

# define the DAG
dag = DAG(
    "Finetune_llm_and_deploy_challenger",
    default_args={"owner": "airflow"},
    description="Finetune and evaluate a LLM model",
    schedule=[Dataset("data_transformed")],
    start_date=datetime(2024, 8, 1),
    tags=["llm"],
    params={
        "llm_base_model": "mistralai/Mistral-7B-Instruct-v0.1",
        "model_id": "viggo-subset-200",
        "train_path": "s3://anyscale-public-materials/llm-finetuning/viggo_inverted/train/subset-200.jsonl", # s3://astronomer-anyscale-demo-2/viggo-data/transformed/joined_data.jsonl
        "valid_path": "s3://anyscale-public-materials/llm-finetuning/viggo_inverted/valid/data.jsonl", # s3://astronomer-anyscale-demo-2/viggo-data/valid/data.jsonl
        "test_path": "s3://anyscale-public-materials/llm-finetuning/viggo_inverted/test/data.jsonl", # s3://astronomer-anyscale-demo-2/viggo-data/test/data.jsonl
        "context_length": 512,
        "num_devices": 4,
        "num_epochs": 12,
        "train_batch_size_per_device": 8,
        "eval_batch_size_per_device": 8,
        "learning_rate": 1e-4,
        "accelerator_type": "A10G",
        "random_seed": 42,
        "sample_size": None,
        "num_few_shot_examples": 0,
        "concurrency": 128,
        "temperature": 0,
        "version": 1,
        "s3_bucket": "astronomer-anyscale-demo-2",
        "s3_key": "'viggo-data/transformed/joined_data.jsonl'",
    },
    render_template_as_native_obj=True,
)

# define the tasks
count_records_task = PythonOperator(
    task_id='record_count',
    python_callable=count_records_in_s3_file,
    op_kwargs={
        'bucket': '{{ params.s3_bucket }}',
        'key': '{{ params.s3_key }}',
    },
    provide_context=True,
    dag=dag,
)

# Task to check record count and decide whether to continue
check_record_count_task = BranchPythonOperator(
    task_id='finetune_threshold_check',
    python_callable=check_record_count,
    dag=dag,
)

# Dummy operator for ending the DAG early
end_dag = DummyOperator(
    task_id='end_dag',
    dag=dag,
)

get_lora_storage_path_task = PythonOperator(
    task_id="get_lora_storage_path",
    python_callable=get_lora_storage_path,
    op_kwargs={
        "version": "{{ params.version }}",
    },
    dag=dag,
)

finetune_cmd = [
    "python",
    "finetune.py",
    "--model-id",
    "{{ params.model_id }}",
    "--llm-base-model",
    "{{ params.llm_base_model }}",
    "--train-path",
    "{{ params.train_path }}",
    "--valid-path",
    "{{ params.valid_path }}",
    "--context-length",
    "{{ params.context_length }}",
    "--num-devices",
    "{{ params.num_devices }}",
    "--num-epochs",
    "{{ params.num_epochs }}",
    "--train-batch-size-per-device",
    "{{ params.train_batch_size_per_device }}",
    "--eval-batch-size-per-device",
    "{{ params.eval_batch_size_per_device }}",
    "--learning-rate",
    "{{ params.learning_rate }}",
    "--accelerator-type",
    "{{ params.accelerator_type }}",
    "--model-storage-path",
    "{{ ti.xcom_pull(task_ids='get_lora_storage_path') }}",
]

finetune_llm_task = SubmitAnyscaleJob(
    task_id="finetune_llm",
    conn_id="anyscale_conn",
    name="llm-finetune",
    entrypoint=" ".join(finetune_cmd),
    image_uri="localhost:5555/anyscale/llm-forge:0.4.2.2",
    compute_config="default-serverless-config:1",
    working_dir=str(DAGS_DIR),
)

build_service_applications_spec_task = PythonOperator(
    task_id="build_service_applications_spec",
    python_callable=build_service_applications_spec,
    op_kwargs={
        "base_model": "{{ params.llm_base_model }}",
        "model_storage_path": "{{ ti.xcom_pull(task_ids='get_lora_storage_path') }}",
    },
    dag=dag,
)

deploy_llm_task = RolloutAnyscaleService(
    task_id="deploy_llm",
    conn_id="anyscale_conn",
    name="finetuned-llm-service",
    working_dir=str(DAGS_DIR),
    image_uri="localhost:5555/anyscale/endpoints_aica:0.5.0-5659",
    compute_config="default-serverless-config:1",
    applications="{{ ti.xcom_pull(task_ids='build_service_applications_spec') }}",
    dag=dag,
)


get_service_url_and_token_task = PythonOperator(
    task_id="get_service_url_and_token",
    python_callable=get_service_url_and_token,
    op_kwargs={
        "service_name": "finetuned-llm-service",
    },
    dag=dag,
)


def terminate_service(service_name: str) -> None:
    anyscale = AnyscaleHook(conn_id="anyscale_conn")
    anyscale.terminate_service(service_name=service_name, time_delay=10)


terminate_service_task = PythonOperator(
    task_id="terminate_service",
    python_callable=terminate_service,
    op_kwargs={
        "service_name": "finetuned-llm-service",
    },
    dag=dag,
)

# define the dependencies
count_records_task >> check_record_count_task
check_record_count_task >> [get_lora_storage_path_task, end_dag]
get_lora_storage_path_task >> finetune_llm_task
finetune_llm_task >> build_service_applications_spec_task
build_service_applications_spec_task >> deploy_llm_task
deploy_llm_task >> get_service_url_and_token_task
get_service_url_and_token_task >> terminate_service_task

# define the default arguments
dag.doc_md = __doc__
dag.doc_md += """
## DAG Parameters
- `llm_base_model`: The base model to use for the LLM.
- `num_devices`: The number of devices to use for training.
- `context_length`: The context length for the LLM.
- `train_batch_size_per_device`: The training batch size per device.
- `eval_batch_size_per_device`: The evaluation batch size per device.
- `learning_rate`: The learning rate for the LLM.
- `accelerator_type`: The accelerator type to use for training.
- `random_seed`: The random seed to use for evaluation.
- `sample_size`: The sample size to use for evaluation.
- `num_few_shot_examples`: The number of few shot examples to use for evaluation.
- `concurrency`: The concurrency to use for evaluation.
- `temperature`: The temperature to use for evaluation.
"""
