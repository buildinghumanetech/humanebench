"""
Custom solver for using pre-generated AI outputs.

This solver reads the pre-generated AI response from the sample metadata
instead of calling a model to generate a new response. This is useful for
scoring existing responses (e.g., golden questions with human ratings).
"""

from inspect_ai.solver import solver, Solver, TaskState, Generate
from inspect_ai.model import ModelOutput


@solver
def use_pregenerated_output() -> Solver:
    """
    Solver that uses pre-generated outputs from dataset metadata.

    This solver checks if the sample metadata contains an 'ai_output' field.
    If present, it uses that as the model output instead of generating a new
    response. If not present, it falls back to normal generation.

    Returns:
        Solver that sets state.output from metadata
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Check if this sample has a pre-generated output in metadata
        # Metadata is nested: state.metadata["metadata"]["ai_output"]
        metadata_dict = state.metadata.get("metadata", {})
        ai_output = metadata_dict.get("ai_output")

        if ai_output:
            # Use the pre-generated output
            state.output = ModelOutput(completion=ai_output)
            state.completed = True
            state.metadata['used_pregenerated_output'] = True
        else:
            # Fall back to normal generation if no pre-generated output
            state = await generate(state)
            state.metadata['used_pregenerated_output'] = False

        return state

    return solve


@solver
def use_pregenerated_output_strict() -> Solver:
    """
    Strict version that requires pre-generated output.

    This solver will raise an error if the sample doesn't have a pre-generated
    output in metadata. Use this when you want to ensure you're only scoring
    pre-existing responses and not accidentally generating new ones.

    Returns:
        Solver that sets state.output from metadata (strict mode)

    Raises:
        ValueError: If no pre-generated output is found in metadata
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Check if this sample has a pre-generated output in metadata
        # Metadata is nested: state.metadata["metadata"]["ai_output"]
        metadata_dict = state.metadata.get("metadata", {})
        ai_output = metadata_dict.get("ai_output")

        if not ai_output:
            raise ValueError(
                f"Sample {state.sample_id} does not have a pre-generated output "
                f"in metadata (expected 'ai_output' field). Use the regular "
                f"use_pregenerated_output() solver if you want to fall back to generation."
            )

        # Use the pre-generated output
        state.output = ModelOutput(completion=ai_output)
        state.completed = True
        state.metadata['used_pregenerated_output'] = True

        return state

    return solve
