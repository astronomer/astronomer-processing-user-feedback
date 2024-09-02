"""Utility functions for deploying Ray LLM applications."""
from anyscale.service.models import ServiceConfig

def get_ray_llm_service_config(
    service_name: str,
    working_dir: str,
    model_config_path: str,
    model_storage_path: str,
    hf_token: str,
) -> ServiceConfig:
    """Build the service config for a Ray LLM application."""
    return ServiceConfig(
        name=service_name,
        working_dir=working_dir,
        image_uri="localhost:5555/anyscale/endpoints_aica:0.5.0-5659",
        compute_config="default-serverless-config:1",
        applications=[
            {
                "name": "llm-endpoint",
                "route_prefix": "/",
                "import_path": "aviary_private_endpoints.backend.server.run:router_application",
                "args": {
                    "models": [],
                    "multiplex_models": [model_config_path],
                    "dynamic_lora_loading_path": model_storage_path,
                },
                "runtime_env": {"env_vars": {"HUGGING_FACE_HUB_TOKEN": hf_token}},
            }
        ],
    )
