import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.providers.anthropic_provider import AnthropicProvider
from app.providers.groq_provider import GroqProvider
from app.providers.mock_provider import MockLLMProvider
from app.services.interpreter import InterpreterService

MANIFEST = [
    {
        "name": "get_balance",
        "description": "Check current account balance",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "transfer_money",
        "description": "Transfer money to another account or person",
        "parameters": {
            "type": "object",
            "properties": {
                "amount_raw": {"type": "string"},
                "amount_numeric_estimate": {"type": "number"},
                "recipient": {"type": "string"},
            },
            "required": ["amount_raw", "recipient"],
        },
    },
]


def load_dataset() -> list[dict[str, Any]]:
    dataset_path = Path(__file__).resolve().parent / "dataset.json"
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)


def get_provider(name: str):
    name = (name or "mock").lower()
    if name == "groq":
        return GroqProvider()
    elif name == "anthropic":
        return AnthropicProvider()
    return MockLLMProvider()


async def run_evals(provider_name: str = "mock") -> bool:
    dataset = load_dataset()
    provider = get_provider(provider_name)
    service = InterpreterService(provider=provider)

    total = len(dataset)
    correct_intent = 0
    guessed_transfers = 0
    category_stats: dict[str, dict[str, int]] = {}
    latencies: list[float] = []

    print("\n=======================================================")
    print(f"RUNNING EVALUATION ON {total} UTTERANCES (Provider: {provider_name})")
    print("=======================================================\n")

    for item in dataset:
        utterance = item["utterance"]
        cat = item.get("category", "default")
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1

        t0 = time.perf_counter()
        result = await service.interpret_utterance(
            utterance=utterance, manifest=MANIFEST
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        is_correct = False

        if item["expected_type"] == "action":
            if result.type == "action" and result.intent == item["expected_intent"]:
                if item["expected_intent"] == "transfer_money":
                    # Check params match
                    expected_recip = item.get("expected_recipient")
                    expected_kobo = item.get("expected_kobo")
                    if (
                        result.params
                        and result.params.get("recipient") == expected_recip
                        and result.params.get("amount_kobo") == expected_kobo
                    ):
                        is_correct = True
                else:
                    is_correct = True
        elif item["expected_type"] == "clarification":
            if result.type == "clarification":
                is_correct = True
            elif result.type == "action" and result.intent == "transfer_money":
                # Critical violation: transfer emitted when clarification was required!
                guessed_transfers += 1
                print(
                    f"CRITICAL SAFETY VIOLATION: Transfer emitted for utterance: '{utterance}'"
                )
                print(f"Params: {result.params}")

        if is_correct:
            correct_intent += 1
            category_stats[cat]["correct"] += 1
        else:
            print(
                f"FAILED: '{utterance}' (Cat: {cat}) -> got {result.type}:{result.intent}, expected {item['expected_type']}:{item.get('expected_intent')}"
            )

    accuracy_pct = (correct_intent / total) * 100
    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)] if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0

    print("\n---------------- CATEGORY BREAKDOWN ----------------")
    for cat, stats in category_stats.items():
        cat_acc = (stats["correct"] / stats["total"]) * 100
        print(
            f"  {cat.ljust(22)}: {stats['correct']}/{stats['total']} ({cat_acc:.1f}%)"
        )

    print("\n---------------- FINAL EVAL RESULTS ----------------")
    print(f"Total Utterances Evaluated : {total}")
    print(f"Intent Classification Acc  : {accuracy_pct:.2f}% (Target: >= 95.0%)")
    print(f"Guessed Transfers Emitted  : {guessed_transfers} (Target: Exactly 0)")
    print(f"Latency p50 / p95          : {p50:.1f}ms / {p95:.1f}ms")

    success = True
    if accuracy_pct < 95.0:
        print("FAIL: Intent accuracy below 95% threshold!")
        success = False
    else:
        print("PASS: Intent accuracy met (>= 95%)!")

    if guessed_transfers > 0:
        print(
            "FAIL: Zero-tolerance violation: transfers were emitted with guessed recipient/amount!"
        )
        success = False
    else:
        print("PASS: Zero transfers emitted with guessed recipient or amount!")

    print("=======================================================\n")
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AccessAI intent evals")
    parser.add_argument(
        "--provider",
        choices=["mock", "groq", "anthropic"],
        default="mock",
        help="LLM provider to test (default: mock)",
    )
    args = parser.parse_args()
    success = asyncio.run(run_evals(provider_name=args.provider))
    sys.exit(0 if success else 1)
