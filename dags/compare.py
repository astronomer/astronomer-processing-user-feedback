"""Compare the evaluation results of two models."""

from typing import Optional

from evaluate import EvaluationResult, get_evaluation_path, resolve_eval_output_path
from pydantic import BaseModel
from typer import Typer


def get_evaluation_results(
    output_path: str,
    model_id: str,
    test_path: str,
    random_seed: int,
    sample_size: Optional[int],
) -> EvaluationResult:
    """Get the evaluation results of a model."""
    eval_path = get_evaluation_path(
        output_path=output_path,
        model_id=model_id,
        test_path=test_path,
        random_seed=random_seed,
        sample_size=sample_size,
    )
    print(f"Reading evaluation results from {eval_path}")
    return EvaluationResult.model_validate_json(eval_path.read_text())


class ComparisonResult(BaseModel):
    champion_eval: EvaluationResult
    challenger_eval: EvaluationResult
    champion_wins: bool


def compare_results(
    champion_eval: EvaluationResult,
    challenger_eval: EvaluationResult,
) -> ComparisonResult:
    """Compare the evaluation results of two models."""
    champion_better_on_fn_accuracy = (
        champion_eval.accuracy_output_fn_pct > challenger_eval.accuracy_output_fn_pct
    )
    champion_better_on_attr_accuracy = (
        champion_eval.accuracy_attr_types_pct > challenger_eval.accuracy_attr_types_pct
    )

    champion_wins = champion_better_on_fn_accuracy or champion_better_on_attr_accuracy
    return ComparisonResult(
        champion_eval=champion_eval,
        challenger_eval=challenger_eval,
        champion_wins=champion_wins,
    )


############################################
# CLI.
############################################
app = Typer()


@app.command()
def compare(
    champion_model_id: str,
    challenger_model_id: str,
    test_path: str,
    random_seed: int,
    sample_size: Optional[int] = None,
    output_dir: Optional[str] = None,
    raise_exception: bool = True
) -> str:
    output_path = resolve_eval_output_path(output_dir)

    champion_eval_results = get_evaluation_results(
        output_path=output_path,
        model_id=champion_model_id,
        test_path=test_path,
        random_seed=random_seed,
        sample_size=sample_size,
    )
    challenger_eval_results = get_evaluation_results(
        output_path=output_path,
        model_id=challenger_model_id,
        test_path=test_path,
        random_seed=random_seed,
        sample_size=sample_size,
    )

    comparison_result = compare_results(champion_eval_results, challenger_eval_results)

    if comparison_result.champion_wins:
        if raise_exception:
            raise ValueError(f"Quality degradation detected: {comparison_result}")
        
        print(f"Quality degradation detected: {comparison_result}")
        return champion_model_id
    else:
        print(f"Quality improvement detected: {comparison_result}")
        return challenger_model_id


if __name__ == "__main__":
    app()
