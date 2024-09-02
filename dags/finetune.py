import os
import shutil
import subprocess
from pathlib import Path

import yaml
import typer
from typer import Typer
from pydantic import BaseModel

DAGS_DIR = Path(__file__).parent


class LLMForgeConfig(BaseModel):
    llm_base_model: str
    train_path: str
    valid_path: str
    context_length: int
    num_devices: int
    num_epochs: int
    train_batch_size_per_device: int
    eval_batch_size_per_device: int
    learning_rate: float
    accelerator_type: str

    def persist(self, config_dir: Path, working_dir: Path):
        # 1. Read the config template
        config_template_path = (
            DAGS_DIR
            / "configs"
            / "finetune"
            / "template"
            / "llm_forge_config_template.yaml"
        )
        with open(config_template_path, "r") as f:
            config_template = f.read()

        # 2. Fill in the template with the provided values
        config = config_template.format(
            model_id=self.llm_base_model,
            train_path=self.train_path,
            valid_path=self.valid_path,
            context_length=self.context_length,
            num_devices=self.num_devices,
            num_epochs=self.num_epochs,
            train_batch_size_per_device=self.train_batch_size_per_device,
            eval_batch_size_per_device=self.eval_batch_size_per_device,
            learning_rate=self.learning_rate,
            accelerator_type=self.accelerator_type,
        )
        # 3. Check if valid yaml
        config_dict = yaml.load(config, Loader=yaml.SafeLoader)

        # 4. Copy the deepspeed config file
        deepspeed_config_path = (
            DAGS_DIR / "configs" / "finetune" / "template" / "deepspeed_config.json"
        )
        config_dir.mkdir(parents=True, exist_ok=True)
        deepseed_dest_path = Path(config_dir) / "deepspeed_config.json"
        shutil.copy(deepspeed_config_path, deepseed_dest_path)
        config_dict["deepspeed"]["config_path"] = str(
            deepseed_dest_path.relative_to(working_dir)
        )

        # 5. Write the filled-in config to a file
        config_file_path = Path(config_dir) / "config.yaml"
        with open(config_file_path, "w") as f:
            yaml.dump(config_dict, f)


def create_finetune_config(
    model_id: str,
    llm_base_model: str,
    train_path: str,
    valid_path: str,
    context_length: int,
    num_devices: int,
    num_epochs: int,
    train_batch_size_per_device: int,
    eval_batch_size_per_device: int,
    learning_rate: float,
    accelerator_type: str,
) -> str:
    full_model_id = f"{llm_base_model}:{model_id}"

    # 1. Create the config file
    llm_forge_config = LLMForgeConfig(
        llm_base_model=llm_base_model,
        train_path=train_path,
        valid_path=valid_path,
        context_length=context_length,
        num_devices=num_devices,
        num_epochs=num_epochs,
        train_batch_size_per_device=train_batch_size_per_device,
        eval_batch_size_per_device=eval_batch_size_per_device,
        learning_rate=learning_rate,
        accelerator_type=accelerator_type,
    )
    config_dir = DAGS_DIR / "configs" / "finetune" / "runs" / full_model_id
    llm_forge_config.persist(config_dir=config_dir, working_dir=DAGS_DIR)
    config_path = config_dir / "config.yaml"
    return str(config_path.relative_to(DAGS_DIR))


app = Typer()


@app.command()
def main(
    model_id: str = typer.Option(
        ..., help="Model ID to be used for storing the fine-tuned model"
    ),
    llm_base_model: str = typer.Option(..., help="Base model to be fine-tuned"),
    train_path: str = typer.Option(..., help="Path to the training dataset"),
    valid_path: str = typer.Option(..., help="Path to the validation dataset"),
    context_length: int = typer.Option(..., help="Context length for the model"),
    num_devices: int = typer.Option(..., help="Number of devices to use for training"),
    num_epochs: int = typer.Option(..., help="Number of epochs for training"),
    train_batch_size_per_device: int = typer.Option(
        ..., help="Training batch size per device"
    ),
    eval_batch_size_per_device: int = typer.Option(
        ..., help="Evaluation batch size per device"
    ),
    learning_rate: float = typer.Option(..., help="Learning rate for training"),
    accelerator_type: str = typer.Option(
        ..., help="Accelerator type to use for training"
    ),
    model_storage_path: str = typer.Option(
        ..., help="LoRA storage URI for storing the fine-tuned model"
    ),
):
    print("Creating fine-tuning config with")
    print(f"model_id: {model_id}")
    print(f"llm_base_model: {llm_base_model}")
    print(f"train_path: {train_path}")
    print(f"valid_path: {valid_path}")
    print(f"context_length: {context_length}")
    print(f"num_devices: {num_devices}")
    print(f"num_epochs: {num_epochs}")
    print(f"train_batch_size_per_device: {train_batch_size_per_device}")
    print(f"eval_batch_size_per_device: {eval_batch_size_per_device}")
    print(f"learning_rate: {learning_rate}")
    print(f"accelerator_type: {accelerator_type}")

    finetune_config_path = create_finetune_config(
        model_id=model_id,
        llm_base_model=llm_base_model,
        train_path=train_path,
        valid_path=valid_path,
        context_length=context_length,
        num_devices=num_devices,
        num_epochs=num_epochs,
        train_batch_size_per_device=train_batch_size_per_device,
        eval_batch_size_per_device=eval_batch_size_per_device,
        learning_rate=learning_rate,
        accelerator_type=accelerator_type,
    )

    entrypoint = f"llmforge dev finetune {finetune_config_path}"

    model_tag = f"{llm_base_model}:{model_id}"
    entrypoint += f" --model-tag={model_tag}"

    lora_storage_uri = model_storage_path
    entrypoint += f" --forward-best-checkpoint-remote-uri={lora_storage_uri}"

    anyscale_artifact_storage = os.environ.get("ANYSCALE_ARTIFACT_STORAGE")

    if anyscale_artifact_storage:
        print(f"ANYSCALE_ARTIFACT_STORAGE is set to {anyscale_artifact_storage}")
    subprocess.run(entrypoint, check=True, shell=True)
    if lora_storage_uri:
        print(
            f"Note: LoRA weights will also be stored in path {lora_storage_uri} under {model_tag} bucket."
        )


if __name__ == "__main__":
    app()
