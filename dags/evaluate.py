"""Evaluate the LLM model on a functional representation task."""

import os
import re
import logging
from hashlib import md5
from pathlib import Path
from typing import Optional, Union, cast, Literal

import numpy as np
import pandas as pd
import ray
import typer
from cloudpathlib import CloudPath
from cloudpathlib.exceptions import InvalidPrefixError
import openai
from openai.resources.chat.completions import ChatCompletionMessageParam
from pydantic import BaseModel
from typer import Typer

############################################
# Logging.
############################################
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

############################################
# Types.
############################################
SystemMessageT = ChatCompletionMessageParam
UserMessageT = ChatCompletionMessageParam
AssistantMessageT = ChatCompletionMessageParam
RolesT = Literal["system", "user", "assistant"]

############################################
# Constants.
############################################
FUNCTION_TYPES = [
    "inform",
    "request",
    "give_opinion",
    "confirm",
    "verify_attribute",
    "suggest",
    "request_explanation",
    "recommend",
    "request_attribute",
]
ATTR_TYPES = [
    "name",
    "exp_release_date",
    "release_year",
    "developer",
    "esrb",
    "rating",
    "genres",
    "player_perspective",
    "has_multiplayer",
    "platforms",
    "available_on_steam",
    "has_linux_release",
    "has_mac_release",
    "specifier",
]
SYSTEM_PROMPT = (
    "Given a target sentence construct the underlying meaning "
    "representation\n"
    "of the input sentence as a single function with attributes and "
    "attribute\n"
    "values. This function should describe the target string "
    "accurately and the\n"
    "function must be one of the following ['inform', 'request', "
    "'give_opinion',\n"
    "'confirm', 'verify_attribute', 'suggest', 'request_explanation',\n"
    "'recommend', 'request_attribute'].\n"
    "\n"
    "The attributes must be one of the following:\n"
    "['name', 'exp_release_date', 'release_year', 'developer', 'esrb', "
    "'rating',\n"
    "'genres', 'player_perspective', 'has_multiplayer', 'platforms',\n"
    "'available_on_steam', 'has_linux_release', 'has_mac_release', "
    "'specifier']\n"
)
ROLES = ["system", "user", "assistant"]

############################################
# Types.
############################################
PathLike = Union[Path, CloudPath]


class ExperimentInfo(BaseModel):
    sample_size: Optional[int]
    random_seed: int
    data_path: str
    llm_model: str
    model_id: str
    num_few_shot_examples: int


############################################
# Data loading.
############################################
def read_data(
    data_path: str,
    sample_size: Optional[int],
    random_seed: int,
) -> tuple[pd.DataFrame, list]:
    df = ray.data.read_json(data_path).to_pandas()
    if sample_size is not None:
        df = df.sample(n=sample_size, replace=False, random_state=random_seed)
    return df


############################################
# Prompt generation.
############################################
def zero_shot(user_input: str) -> tuple[SystemMessageT, UserMessageT]:
    """Return a zero-shot completion of the user input."""
    return (
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_input,
        },
    )


def few_shot(
    user_input: str,
    examples: list[
        tuple[
            SystemMessageT,
            UserMessageT,
            AssistantMessageT,
        ]
    ],
) -> tuple[SystemMessageT, UserMessageT]:
    """Build a prompt for few-shot learning given a user input and examples."""
    system_message, user_message = zero_shot(user_input)
    user_text = user_message["content"]

    example_preface = (
        "Examples are printed below."
        if len(examples) > 1
        else "An example is printed below."
    )
    example_preface += (
        ' Note: you are to respond with the string after "Output: " only.'
    )
    examples_parsed = "\n".join(
        [
            f"{user['content']}\nOutput: {assistant['content']}"
            for (system, user, assistant) in examples
        ]
    )
    response_preface = "Now please provide the output for:\n"
    user_text = f"{example_preface}\n{examples_parsed}\n{response_preface}\n{user_text}\nOutput: "
    return (system_message, {"role": "user", "content": user_text})


############################################
# Data extraction.
############################################
def extract_role(
    messages: tuple[
        SystemMessageT,
        UserMessageT,
        AssistantMessageT,
    ],
    role: RolesT,
) -> str:
    assert len(messages) == 3
    message = messages[ROLES.index(role)]
    assert message["role"] == role
    return cast(str, message["content"])


def parse_messages(record: dict) -> dict:
    return {
        "user_input": extract_role(record["messages"], role="user"),
        "ground_truth": extract_role(record["messages"], role="assistant"),
    }


def generate_inputs(
    record: dict,
    examples: list[list[SystemMessageT, UserMessageT, AssistantMessageT]],
) -> dict:
    if len(examples) == 0:
        sys, usr = zero_shot(record["user_input"])
    elif len(examples) >= 1:
        sys, usr = few_shot(
            user_input=record["user_input"],
            examples=examples,
        )
    record["model_input"] = usr["content"]
    return record


############################################
# Response generation.
############################################
def build_client(base_url: str, api_key: str, canary_deployment: bool) -> openai.OpenAI:
    if canary_deployment:
        default_headers = {"X-Anyscale-Version": "canary"}
    else:
        default_headers = None
    return openai.OpenAI(
        base_url=base_url.rstrip("/") + "/v1",
        api_key=api_key,
        default_headers=default_headers,
    )


def query(
    client: openai.OpenAI,
    llm_model: str,
    system_message: SystemMessageT,
    user_message: UserMessageT,
    temperature: float = 0,
    timeout: float = 3 * 60,
) -> Optional[str]:
    model_response = client.chat.completions.create(
        model=llm_model,
        messages=[system_message, user_message],
        temperature=temperature,
        timeout=timeout,
    )
    model_output = model_response.choices[0].message.content
    return model_output


class ModelClient:
    def __init__(self, base_url: str, api_key: str, canary_deployment: bool) -> None:
        """Initialize the model client."""
        self.client = build_client(base_url, api_key, canary_deployment)

    def __call__(
        self,
        record: dict,
        llm_model: str,
        temperature: float,
    ) -> dict:
        """Query the model and return the response."""
        try:
            record["model_output"] = query(
                client=self.client,
                llm_model=llm_model,
                system_message={"role": "system", "content": SYSTEM_PROMPT},
                user_message={"role": "user", "content": record["model_input"]},
                temperature=temperature,
            )

        except Exception as e:
            record["model_output"] = ""
            record["model_failure"] = str(e)

        else:
            record["model_failure"] = np.nan

        return record


############################################
# Post-processing of responses.
############################################
def post_process(record: dict) -> dict:
    record.update(
        {
            "ground_truth_fn_type": extract_function_type(record["ground_truth"]),
            "ground_truth_attr_types": extract_attribute_types(record["ground_truth"]),
            "model_fn_type": extract_function_type(record["model_output"]),
            "model_attr_types": extract_attribute_types(record["model_output"]),
        }
    )
    return record


def generate_outputs(
    data_path: str,
    query_url: str,
    query_auth_token: str,
    random_seed: int,
    sample_size: Optional[int],
    llm_model: str,
    examples: list[list[SystemMessageT, UserMessageT, AssistantMessageT]],
    concurrency: int,
    temperature: float,
    canary_deployment: bool,
) -> pd.DataFrame:
    df = read_data(
        data_path=data_path,
        random_seed=random_seed,
        sample_size=sample_size,
    )
    print(f"Data loaded with {df.shape[0]} examples")
    print(f"Number of few shot examples {len(examples)}")

    df_inputs_outputs = (
        ray.data.from_pandas(df)
        .repartition(concurrency)
        .map(parse_messages, num_cpus=0.01)
        .map(generate_inputs, fn_kwargs={"examples": examples}, num_cpus=0.01)
        .map(
            ModelClient,
            concurrency=concurrency,
            num_cpus=0.01,
            fn_kwargs={
                "llm_model": llm_model,
                "temperature": temperature,
            },
            fn_constructor_kwargs={
                "base_url": query_url,
                "api_key": query_auth_token,
                "canary_deployment": canary_deployment,
            },
        )
        .map(post_process)
        .to_pandas()
    )

    return df_inputs_outputs


############################################
# Evaluation.
############################################
def resolve_eval_output_path(output_dir: Optional[str]) -> PathLike:
    if output_dir is not None:
        try:
            output_path = CloudPath(output_dir)  # type: ignore
        except InvalidPrefixError:
            output_path = Path(output_dir)  # type: ignore
    else:
        artifact_storage_eval = (
            CloudPath(os.environ["ANYSCALE_ARTIFACT_STORAGE"]) / "evaluation"  # type: ignore
        )
        output_path = artifact_storage_eval
    return output_path


def get_experiment_dir(
    output_path: PathLike,
    model_id: str,
    data_path: str,
    random_seed: int,
    sample_size: Optional[int],
) -> PathLike:
    data_attrs = {"data_path": data_path, "sample_size": sample_size}
    if sample_size is not None:
        data_attrs["random_seed"] = random_seed
    data_attrs_str = "_".join(f"{key}_{value}" for key, value in data_attrs.items())
    data_hash = md5(data_attrs_str.encode()).hexdigest()
    return output_path / model_id / data_hash


def get_evaluation_path(
    output_path: PathLike,
    model_id: str,
    test_path: str,
    random_seed: int,
    sample_size: Optional[int],
) -> PathLike:
    experiment_dir = get_experiment_dir(
        output_path=output_path,
        model_id=model_id,
        data_path=test_path,
        random_seed=random_seed,
        sample_size=sample_size,
    )
    return experiment_dir / "evaluation_result.json"


def get_experiment_info_path(
    output_path: PathLike,
    model_id: str,
    test_path: str,
    random_seed: int,
    sample_size: Optional[int],
) -> PathLike:
    experiment_dir = get_experiment_dir(
        output_path=output_path,
        model_id=model_id,
        data_path=test_path,
        random_seed=random_seed,
        sample_size=sample_size,
    )
    return experiment_dir / "experiment_info.json"


class EvaluationResult(BaseModel):
    failed_api_responses_pct: float
    accuracy_format_fn_type_pct: float
    accuracy_format_attr_types_pct: float
    accuracy_output_fn_pct: float
    accuracy_attr_types_pct: float
    accuracy_attr_types_order_pct: float


def extract_function_type(response: Optional[str]) -> Optional[str]:
    """Extract the function type from the response."""
    if response is None:
        return None

    # pattern to match is "{function_type}({attributes})"
    expected_pattern = re.compile(r"^(?P<function_type>.+?)\((?P<attributes>.+)\)$")

    # remove any "Output: " prefix and strip the response
    match = expected_pattern.match(response.split("Output: ")[-1].strip())

    if match is None:
        return None

    # return the function type
    ret = match.group("function_type")
    return ret.replace("\\_", "_")  # handle escapes of underscores


def extract_attribute_types(response: Optional[str]) -> list[str]:
    if response is None:
        return []

    # pattern to match is "{function_type}({attributes})"
    expected_pattern = re.compile(r"^(?P<function_type>.+?)\((?P<attributes>.+)\)$")

    # remove any "Output: " prefix and strip the response
    match = expected_pattern.match(response.split("Output: ")[-1].strip())

    if match is None:
        return []

    attributes = match.group("attributes")

    # pattern is "{attribute_type}[{attribute_value}], ..."
    attr_types = re.findall(r"(\w+)\[", attributes)

    return attr_types


def evaluate_experiment(df: pd.DataFrame) -> EvaluationResult:
    # we will get a few bad responses due to service unreliability
    failed_api_responses = df["model_failure"].notnull()
    failed_api_responses_pct = failed_api_responses.mean() * 100

    # we will evaluate the accuracy of the model on the good responses
    df = df[~failed_api_responses]

    # first we will check if the format of the answer is correct
    correct_format_model_fn_type = df["model_fn_type"].notnull()
    accuracy_format_fn_type_pct = correct_format_model_fn_type.mean() * 100

    correct_format_model_attr_types = df["model_attr_types"].apply(len) > 0
    accuracy_format_attr_types_pct = correct_format_model_attr_types.mean() * 100

    # second we will check if the answer captures the function type
    correct_fn_type = df["ground_truth_fn_type"] == df["model_fn_type"]
    accuracy_output_fn_pct = correct_fn_type.mean() * 100

    # third we will check if the answer captures the attribute types
    correct_attr_types = df["ground_truth_attr_types"].apply(set) == df[
        "model_attr_types"
    ].apply(set)
    accuracy_attr_types_pct = correct_attr_types.mean() * 100

    # fourth we will check if the answer captures the attribute types in the correct order
    correct_attr_types_order = df["ground_truth_attr_types"].apply(list) == df[
        "model_attr_types"
    ].apply(list)
    accuracy_attr_types_order_pct = correct_attr_types_order.mean() * 100

    return EvaluationResult(
        failed_api_responses_pct=failed_api_responses_pct,
        accuracy_format_fn_type_pct=accuracy_format_fn_type_pct,
        accuracy_format_attr_types_pct=accuracy_format_attr_types_pct,
        accuracy_output_fn_pct=accuracy_output_fn_pct,
        accuracy_attr_types_pct=accuracy_attr_types_pct,
        accuracy_attr_types_order_pct=accuracy_attr_types_order_pct,
    )


############################################
# CLI.
############################################
app = Typer()


@app.command()
def run(
    data_path: str,
    query_url: str,
    query_auth_token: str,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
    llm_model: str = "base",
    num_few_shot_examples: int = 0,
    few_shot_data_path: Optional[str] = None,
    concurrency: int = 256,
    output_dir: Optional[str] = None,
    temperature: float = 0.0,
    check_existing: bool = False,
    canary_deployment: bool = False,
) -> None:
    output_path = resolve_eval_output_path(output_dir)

    # 1. Record the experiment info.
    experiment_info = ExperimentInfo(
        sample_size=sample_size,
        random_seed=random_seed,
        data_path=data_path,
        llm_model=llm_model,
        model_id=llm_model,
        num_few_shot_examples=num_few_shot_examples,
    )

    experiment_dir = get_experiment_dir(
        output_path=output_path,
        model_id=llm_model,
        data_path=data_path,
        random_seed=random_seed,
        sample_size=sample_size,
    )
    if check_existing and experiment_dir.exists():
        typer.confirm(
            f"Experiment already exists at {experiment_dir} - do you want to overwrite?"
        )

    experiment_dir.mkdir(exist_ok=True, parents=True)
    experiment_info_metadata_path = get_experiment_info_path(
        output_path=output_path,
        model_id=llm_model,
        test_path=data_path,
        random_seed=random_seed,
        sample_size=sample_size,
    )
    logger.info(f"Saving experiment info to {experiment_info_metadata_path}")
    experiment_info_metadata_path.write_text(experiment_info.model_dump_json())

    # 2. Run the experiment.
    logger.info("Running the experiment")
    examples = []
    if num_few_shot_examples > 0:
        few_shot_df = read_data(
            data_path=few_shot_data_path,
            random_seed=random_seed,
            sample_size=num_few_shot_examples,
        )
        examples = few_shot_df["messages"].tolist()

    df_inputs_outputs = generate_outputs(
        query_url=query_url,
        query_auth_token=query_auth_token,
        data_path=data_path,
        random_seed=random_seed,
        sample_size=sample_size,
        llm_model=llm_model,
        examples=examples,
        concurrency=concurrency,
        temperature=temperature,
        canary_deployment=canary_deployment,
    )

    # 3. Save the input/outputs.
    input_outputs_path = experiment_dir / "input_outputs.jsonl"
    input_outputs_path_str = str(input_outputs_path)
    logger.info(f"Saving inputs/outputs to {input_outputs_path_str}")
    df_inputs_outputs.to_json(input_outputs_path_str, orient="records", lines=True)

    # 4. Evaluate the experiment.
    logger.info("Evaluating the experiment")
    evaluation_result = evaluate_experiment(df=df_inputs_outputs)
    evaluation_result_path = get_evaluation_path(
        output_path=output_path,
        model_id=llm_model,
        test_path=data_path,
        random_seed=random_seed,
        sample_size=sample_size,
    )
    logger.info(f"Saving evaluation result to {evaluation_result_path}")
    evaluation_result_path.write_text(evaluation_result.model_dump_json())


if __name__ == "__main__":
    app()
