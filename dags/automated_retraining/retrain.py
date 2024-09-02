# """Retraining DAG for LLM finetuning and evaluation."""

import os
from pathlib import Path
from typing import Optional
from anyscale_provider.operators.anyscale import (
    SubmitAnyscaleJob,
    RolloutAnyscaleService,
)
from anyscale_provider.hooks.anyscale import AnyscaleHook
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.datasets import Dataset
from airflow.utils.state import State

DAGS_DIR = Path(__file__).parent.parent


def get_lora_storage_path(version: int) -> str:
    """Get the path to the LORA storage."""
    bucket = os.environ["BUCKET"]
    org_id = os.environ["ORG_ID"]
    cloud_id = os.environ["CLOUD_ID"]

    return f"s3://{bucket}/{org_id}/{cloud_id}/artifact_storage/lora_fine_tuning/v{version}/"


def build_service_applications_spec(
    base_model: str,
    model_storage_path: str,
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
        "--canary-deployment",
    ]
    if sample_size:
        cmd.extend(["--sample-size", str(sample_size)])
    return " ".join(cmd)


def build_process_evaluation_results_command(
    llm_base_model: str,
    champion_model_id: str,
    challenger_model_id: str,
    test_path: str,
    random_seed: int,
    sample_size: Optional[int],
) -> bool:
    """Submit a job that performs the comparison between the champion and challenger models."""
    cmd = [
        "python",
        "compare.py",
        f"{llm_base_model}:{champion_model_id}",
        f"{llm_base_model}:{challenger_model_id}",
        test_path,
        str(random_seed),
    ]
    if sample_size:
        cmd.extend(["--sample-size", str(sample_size)])

    return " ".join(cmd)


def alert_for_quality_degradation():
    raise ValueError("Quality degradation detected. Alerting stakeholders.")


def decide_on_deploy(**kwargs):
    dag_instance = kwargs["dag"]
    execution_date = kwargs["execution_date"]

    processed_state = dag_instance.get_task("process-evaluation-results.process_evaluation_results")
    task_instance = TaskInstance(processed_state, execution_date)
    processed_task_state = task_instance.current_state()

    if processed_task_state == State.SUCCESS:
        return "rollout-challenger.deploy_challenger_llm"
    else:
        return "alert-degradation.alert_for_quality_degradation"


with DAG(
    "llm_retrain_and_rollout",
    default_args={"owner": "airflow"},
    description="Retrain an LLM model",
    schedule=None,
    tags=["llm"],
    params={
        "llm_base_model": "mistralai/Mistral-7B-Instruct-v0.1",
        "champion_model_id": "viggo-subset-200",
        "challenger_model_id": "viggo-subset-500",
        "train_path": "s3://anyscale-public-materials/llm-finetuning/viggo_inverted/train/subset-500.jsonl",
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
        "version": 2,
    },
    render_template_as_native_obj=True,
) as dag:

    with TaskGroup("finetune-challenger") as finetune_challenger_llm_task_group:
        get_lora_storage_path_task = PythonOperator(
            task_id="get_lora_storage_path",
            python_callable=get_lora_storage_path,
            op_kwargs={
                "version": "{{ params.version }}",
            },
        )

        finetune_cmd = [
            "python",
            "finetune.py",
            "--model-id",
            "{{ params.challenger_model_id }}",
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
            "{{ ti.xcom_pull(task_ids='finetune-challenger.get_lora_storage_path') }}",
        ]

        finetune_challenger_llm_task = SubmitAnyscaleJob(
            task_id="finetune_challenger_llm",
            conn_id="anyscale_conn",
            name="llm-finetune",
            entrypoint=" ".join(finetune_cmd),
            image_uri="localhost:5555/anyscale/llm-forge:0.4.2.2",
            working_dir=str(DAGS_DIR),
            dag=dag,
        )
        get_lora_storage_path_task >> finetune_challenger_llm_task

    with TaskGroup(
        "deploy-challenger-silent"
    ) as deploy_challenger_silent_llm_task_group:
        build_service_applications_spec_task = PythonOperator(
            task_id="build_service_applications_spec",
            python_callable=build_service_applications_spec,
            op_kwargs={
                "base_model": "{{ params.llm_base_model }}",
                "model_storage_path": "{{ ti.xcom_pull(task_ids='finetune-challenger.get_lora_storage_path') }}",
            },
        )

        silent_deploy_challenger_llm_task = RolloutAnyscaleService(
            task_id="silent_deploy_challenger_llm",
            conn_id="anyscale_conn",
            name="finetuned-llm-service",
            working_dir=str(DAGS_DIR),
            image_uri="localhost:5555/anyscale/endpoints_aica:0.5.0-5659",
            applications="{{ ti.xcom_pull(task_ids='deploy-challenger-silent.build_service_applications_spec') }}",
            canary_percent=0,
        )

        build_service_applications_spec_task >> silent_deploy_challenger_llm_task

    with TaskGroup("evaluate-challenger") as evaluate_challenger_llm_task_group:
        get_service_url_and_token_task = PythonOperator(
            task_id="get_service_url_and_token",
            python_callable=get_service_url_and_token,
            op_kwargs={
                "service_name": "finetuned-llm-service",
            },
        )

        build_evaluate_command_task = PythonOperator(
            task_id="build_evaluate_command",
            python_callable=build_evaluate_command,
            op_kwargs={
                "evaluation_data_path": "{{ params.test_path }}",
                "query_url": "{{ ti.xcom_pull(task_ids='evaluate-challenger.get_service_url_and_token')[0] }}",
                "query_auth_token": "{{ ti.xcom_pull(task_ids='evaluate-challenger.get_service_url_and_token')[1] }}",
                "random_seed": "{{ params.random_seed }}",
                "sample_size": "{{ params.sample_size }}",
                "base_model_id": "{{ params.llm_base_model }}",
                "model_id": "{{ params.challenger_model_id }}",
                "num_few_shot_examples": "{{ params.num_few_shot_examples }}",
                "concurrency": "{{ params.concurrency }}",
                "temperature": "{{ params.temperature }}",
            },
        )

        evaluate_challenger_llm_task = SubmitAnyscaleJob(
            task_id="evaluate_challenger_llm",
            conn_id="anyscale_conn",
            name="llm-evaluate",
            entrypoint="{{ ti.xcom_pull(task_ids='evaluate-challenger.build_evaluate_command') }}",
            image_uri="anyscale/image/ray-plus-openai-cloudpathlib:1",
            working_dir=str(DAGS_DIR),
        )

        (
            get_service_url_and_token_task
            >> build_evaluate_command_task
            >> evaluate_challenger_llm_task
        )

    with TaskGroup(
        "process-evaluation-results"
    ) as process_evaluation_results_task_group:
        build_process_evaluation_results_command_task = PythonOperator(
            task_id="build_process_evaluation_results_command",
            python_callable=build_process_evaluation_results_command,
            op_kwargs={
                "llm_base_model": "{{ params.llm_base_model }}",
                "champion_model_id": "{{ params.champion_model_id }}",
                "challenger_model_id": "{{ params.challenger_model_id }}",
                "test_path": "{{ params.test_path }}",
                "random_seed": "{{ params.random_seed }}",
                "sample_size": "{{ params.sample_size }}",
            },
        )

        process_evaluation_results_task = SubmitAnyscaleJob(
            task_id="process_evaluation_results",
            conn_id="anyscale_conn",
            name="llm-compare",
            entrypoint="{{ ti.xcom_pull(task_ids='process-evaluation-results.build_process_evaluation_results_command') }}",
            image_uri="anyscale/image/ray-plus-openai-cloudpathlib:1",
            working_dir=str(DAGS_DIR),
        )

        build_process_evaluation_results_command_task >> process_evaluation_results_task

    deploy_decision = BranchPythonOperator(
        task_id="decide_on_deploy",
        provide_context=True,
        python_callable=decide_on_deploy,
    )

    with TaskGroup("rollout-challenger") as rollout_challenger_llm_task_group:
        deploy_challenger_llm_task = RolloutAnyscaleService(
            task_id="deploy_challenger_llm",
            conn_id="anyscale_conn",
            name="finetuned-llm-service",
            image_uri="localhost:5555/anyscale/endpoints_aica:0.5.0-5659",
            applications="{{ ti.xcom_pull(task_ids='deploy-challenger-silent.build_service_applications_spec') }}",
            canary_percent=100,
        )

    with TaskGroup("alert-degradation") as alert_degradation_task_group:
        alert_for_quality_degradation_task = PythonOperator(
            task_id="alert_for_quality_degradation",
            python_callable=alert_for_quality_degradation,
        )

    (
        finetune_challenger_llm_task_group
        >> deploy_challenger_silent_llm_task_group
        >> evaluate_challenger_llm_task_group
        >> process_evaluation_results_task_group
        >> deploy_decision
    )
    deploy_decision >> rollout_challenger_llm_task_group
    deploy_decision >> alert_degradation_task_group
