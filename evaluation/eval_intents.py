"""
evaluate_intents.py

Evaluates the NLU module's intent classification performance.

Metrics:
- Accuracy: fraction of correctly classified intents
- Per-class Precision, Recall, F1 (macro and weighted averages)
- Confusion matrix

Usage:
    python evaluate_intents.py \
        --test-file test_intents.json \
        --model-name qwen3 \
        --device cuda:0
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main_CarAdvisor import nlu as NLU_SYSTEM_PROMPT

from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import torch
from transformers import AutoTokenizer

from utils import MODELS, NLU

def load_test_cases(path: str):
    with open(path) as f:
        return json.load(f)


def run_evaluation(args):
    test_cases = load_test_cases(args.test_file)

    # ── Load model ──────────────────────────────────────────────────────────
    model_name, InitModel, prepare_text = MODELS[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = InitModel(model_name, dtype="auto")
    model.eval()

    nlu_engine = NLU(model, tokenizer, prepare_text, NLU_SYSTEM_PROMPT)

    # ── Run inference ───────────────────────────────────────────────────────
    y_true = []
    y_pred = []
    errors = []

    for tc in test_cases:
        test_id = tc["id"]
        user_input = tc["user_input"]
        expected_intent = tc["expected_intent"]

        try:
            state = nlu_engine.parse(user_input, [], args.n_exchanges)
            predicted_intent = state.get("intent", "PARSE_ERROR")
        except Exception as e:
            predicted_intent = "PARSE_ERROR"
            errors.append({"id": test_id, "error": str(e)})

        y_true.append(expected_intent)
        y_pred.append(predicted_intent)

        if args.verbose:
            match = "✓" if predicted_intent == expected_intent else "✗"
            print(f"[{match}] {test_id}: expected={expected_intent}, got={predicted_intent}")

        # Free CUDA memory after each call
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Compute metrics ─────────────────────────────────────────────────────
    all_labels = sorted(set(y_true + y_pred))

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=all_labels, zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", labels=all_labels, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INTENT CLASSIFICATION EVALUATION")
    print("=" * 60)
    print(f"Total samples : {len(y_true)}")
    print(f"Parse errors  : {len(errors)}")
    print(f"\nAccuracy      : {accuracy:.4f}  ({sum(a==b for a,b in zip(y_true,y_pred))}/{len(y_true)})")
    print(f"\nMacro avg     → Precision: {precision_macro:.4f}  Recall: {recall_macro:.4f}  F1: {f1_macro:.4f}")
    print(f"Weighted avg  → Precision: {precision_weighted:.4f}  Recall: {recall_weighted:.4f}  F1: {f1_weighted:.4f}")

    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

    print("Confusion matrix (rows=true, cols=predicted):")
    header = "  ".join(f"{l[:10]:>12}" for l in all_labels)
    print(f"{'':>20}  {header}")
    for label, row in zip(all_labels, cm):
        row_str = "  ".join(f"{v:>12}" for v in row)
        print(f"{label[:20]:>20}  {row_str}")

    if errors:
        print("\nParse errors:")
        for e in errors:
            print(f"  {e['id']}: {e['error']}")

    # ── Save results to JSON ─────────────────────────────────────────────────
    results = {
        "accuracy": accuracy,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted,
        "parse_errors": errors,
        "per_sample": [
            {"id": tc["id"], "expected": e, "predicted": p}
            for tc, e, p in zip(test_cases, y_true, y_pred)
        ],
    }

    out_path = args.output or "intent_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NLU intent classification.")
    parser.add_argument("--test-file", default="test_intents.json")
    parser.add_argument("--model-name", choices=MODELS.keys(), default="qwen3")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--n-exchanges", type=int, default=10)
    parser.add_argument("--output", type=str, default=None, help="Path for JSON output")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())