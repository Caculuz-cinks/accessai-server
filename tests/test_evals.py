import pytest

from tests.evals.run_evals import run_evals


@pytest.mark.asyncio
async def test_eval_benchmark_accuracy_and_safety():
    success = await run_evals(provider_name="mock")
    assert success is True, (
        "Evaluation benchmark failed to achieve 95%+ accuracy or violated zero-guessing policy"
    )
