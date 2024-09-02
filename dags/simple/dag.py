"""Simple DAG for LLM finetuning and evaluation."""

import os
from pathlib import Path
from typing import Optional
from pendulum import datetime
from anyscale_provider.operators.anyscale import (
    SubmitAnyscaleJob,
    RolloutAnyscaleService,
)
from anyscale_provider.hooks.anyscale import AnyscaleHook
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.datasets import Dataset

DAGS_DIR = Path(__file__).parent.parent


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


def build_evaluate_command(
    evaluation_data_path: str,
    query_url: str,
    query_auth_token: str,
    random_seed: int,
    sample_size: Optional[int],
    base_model_id: str,
    model_id: str,
    num_few_shot_examples: int,
    concurrency: int,
    temperature: float,
) -> str:
    cmd = [
        "python",
        "evaluate.py",
        evaluation_data_path,
        query_url,
        query_auth_token,
        "--random-seed",
        str(random_seed),
        "--llm-model",
        f"{base_model_id}:{model_id}",
        "--num-few-shot-examples",
        str(num_few_shot_examples),
        "--concurrency",
        str(concurrency),
        "--temperature",
        str(temperature),
    ]
    if sample_size:
        cmd.extend(["--sample-size", str(sample_size)])
    return " ".join(cmd)


# define the DAG
dag = DAG(
    "llm_finetune_and_evaluate",
    default_args={"owner": "airflow"},
    description="Finetune and evaluate a LLM model",
    schedule=None,
    start_date=datetime(2024, 8, 1),
    tags=["llm"],
    params={
        "llm_base_model": "mistralai/Mistral-7B-Instruct-v0.1",
        "model_id": "viggo-subset-200",
        "train_path": "s3://anyscale-public-materials/llm-finetuning/viggo_inverted/train/subset-200.jsonl",
        "valid_path": "s3://anyscale-public-materials/llm-finetuning/viggo_inverted/valid/data.jsonl",
        "test_path": "s3://anyscale-public-materials/llm-finetuning/viggo_inverted/test/data.jsonl",
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
    },
    render_template_as_native_obj=True,
)

# define the tasks
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


evaluate_llm_command_build_task = PythonOperator(
    task_id="evaluate_llm_command_build",
    python_callable=build_evaluate_command,
    op_kwargs={
        "evaluation_data_path": "{{ params.test_path }}",
        "query_url": "{{ ti.xcom_pull(task_ids='get_service_url_and_token')[0] }}",
        "query_auth_token": "{{ ti.xcom_pull(task_ids='get_service_url_and_token')[1] }}",
        "random_seed": "{{ params.random_seed }}",
        "sample_size": "{{ params.sample_size }}",
        "base_model_id": "{{ params.llm_base_model }}",
        "model_id": "{{ params.model_id }}",
        "num_few_shot_examples": "{{ params.num_few_shot_examples }}",
        "concurrency": "{{ params.concurrency }}",
        "temperature": "{{ params.temperature }}",
    },
    dag=dag,
)

evaluate_llm_task = SubmitAnyscaleJob(
    task_id="evaluate_llm",
    conn_id="anyscale_conn",
    name="evaluate-llm-{{ params.model_id }}",
    entrypoint="{{ ti.xcom_pull(task_ids='evaluate_llm_command_build') }}",
    image_uri="anyscale/image/ray-plus-openai-cloudpathlib:1",
    compute_config="default-serverless-config:1",
    working_dir=str(DAGS_DIR),
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
get_lora_storage_path_task >> finetune_llm_task
finetune_llm_task >> build_service_applications_spec_task
build_service_applications_spec_task >> deploy_llm_task
deploy_llm_task >> get_service_url_and_token_task
get_service_url_and_token_task >> evaluate_llm_command_build_task
evaluate_llm_command_build_task >> evaluate_llm_task
evaluate_llm_task >> terminate_service_task

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
