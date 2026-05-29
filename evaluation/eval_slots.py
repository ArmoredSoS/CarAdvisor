"""
evaluate_slots.py

Evaluates the NLU module's slot filling performance.

For each test case the script compares the predicted slot values against the
ground truth, treating EACH (test_case, slot) pair as one binary sample:
- True Positive  (TP): slot expected non-null and correctly extracted
- False Positive (FP): slot predicted non-null but expected null (hallucination)
- False Negative (FN): slot expected non-null but not extracted (missed)
- True Negative  (TN): both expected and predicted are null

Metrics reported:
- Per-slot Precision / Recall / F1
- Micro-averaged (global) Precision / Recall / F1
- Exact-match accuracy per test case (all non-null slots must match)

Numeric slots (car_price) use a tolerance window (default ±10 %).
String slots use a case-insensitive comparison with optional substring matching.

Usage:
    python evaluate_slots.py \
        --test-file test_slots.json \
        --model-name qwen3 \
        --device cuda:0
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main_CarAdvisor import nlu as NLU_SYSTEM_PROMPT

from collections import defaultdict
import torch
from transformers import AutoTokenizer

from utils import MODELS, NLU

ALL_SLOTS = [
    "car_type", "car_price", "car_state", "car_usecase",
    "fuel_type", "car_brand", "car_dimensions", "fuel_efficiency", "car_design",
]

NULL_VALUES = {None, "null", ""}


def is_null(v):
    if v is None:
        return True
    if isinstance(v, str) and v.lower().strip() in ("null", ""):
        return True
    return False


def slots_match(slot: str, predicted, expected, price_tolerance: float = 0.20) -> bool:
    """Return True if the predicted value matches the expected value for a slot."""
    if is_null(expected) and is_null(predicted):
        return True
    if is_null(expected) or is_null(predicted):
        return False

    if slot == "car_price":
        try:
            pred_v = int(float(str(predicted).replace(",", "")))
            exp_v = int(float(str(expected).replace(",", "")))
            return abs(pred_v - exp_v) <= price_tolerance * exp_v
        except (ValueError, TypeError):
            return False

    # String comparison: case-insensitive, allow substring
    pred_str = str(predicted).lower().strip()
    exp_str = str(expected).lower().strip()
    return pred_str == exp_str or exp_str in pred_str or pred_str in exp_str


def load_test_cases(path: str):
    with open(path) as f:
        return json.load(f)


def run_evaluation(args):
    test_cases = load_test_cases(args.test_file)

    model_name, InitModel, prepare_text = MODELS[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = InitModel(model_name, dtype="auto")
    model.eval()

    nlu_engine = NLU(model, tokenizer, prepare_text, NLU_SYSTEM_PROMPT)

    # Per-slot counters
    slot_tp = defaultdict(int)
    slot_fp = defaultdict(int)
    slot_fn = defaultdict(int)
    slot_tn = defaultdict(int)

    exact_matches = 0
    sample_results = []
    parse_errors = []

    for tc in test_cases:
        test_id = tc["id"]
        user_input = tc["user_input"]
        expected_slots = {k: v for k, v in tc["expected_slots"].items()}

        try:
            state = nlu_engine.parse(user_input, [], args.n_exchanges)
            predicted_slots = state.get("slots", {})
        except Exception as e:
            predicted_slots = {}
            parse_errors.append({"id": test_id, "error": str(e)})

        case_correct = True
        case_details = {"id": test_id, "slots": {}}

        for slot in ALL_SLOTS:
            exp = expected_slots.get(slot)
            pred = predicted_slots.get(slot)

            exp_null = is_null(exp)
            pred_null = is_null(pred)

            if not exp_null and not pred_null:
                if slots_match(slot, pred, exp, args.price_tolerance):
                    slot_tp[slot] += 1
                    case_details["slots"][slot] = "TP"
                else:
                    # Wrong value: counts as FN (missed correct) + FP (wrong value emitted)
                    slot_fn[slot] += 1
                    slot_fp[slot] += 1
                    case_correct = False
                    case_details["slots"][slot] = f"WRONG (exp={exp}, got={pred})"
            elif not exp_null and pred_null:
                slot_fn[slot] += 1
                case_correct = False
                case_details["slots"][slot] = f"FN (exp={exp})"
            elif exp_null and not pred_null:
                slot_fp[slot] += 1
                case_correct = False
                case_details["slots"][slot] = f"FP (hallucinated={pred})"
            else:
                slot_tn[slot] += 1
                case_details["slots"][slot] = "TN"

        if case_correct:
            exact_matches += 1

        if args.verbose:
            match_str = "✓" if case_correct else "✗"
            print(f"\n[{match_str}] {test_id}: {user_input}")
            for slot, outcome in case_details["slots"].items():
                if outcome not in ("TN",):
                    print(f"     {slot}: {outcome}")

        sample_results.append(case_details)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Compute per-slot metrics ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SLOT FILLING EVALUATION")
    print("=" * 70)
    print(f"Total samples: {len(test_cases)}  |  Parse errors: {len(parse_errors)}")
    print(f"Exact-match accuracy (all non-null slots): "
          f"{exact_matches}/{len(test_cases)} = {exact_matches/len(test_cases):.4f}\n")

    print(f"{'Slot':<20} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
          f"{'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 80)

    total_tp = total_fp = total_fn = 0

    per_slot_metrics = {}
    for slot in ALL_SLOTS:
        tp = slot_tp[slot]
        fp = slot_fp[slot]
        fn = slot_fn[slot]
        tn = slot_tn[slot]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        per_slot_metrics[slot] = {"precision": precision, "recall": recall, "f1": f1,
                                  "tp": tp, "fp": fp, "fn": fn, "tn": tn}

        print(f"{slot:<20} {tp:>5} {fp:>5} {fn:>5} {tn:>5}  "
              f"{precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")

        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1        = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                       if (micro_precision + micro_recall) > 0 else 0.0)

    print("-" * 80)
    print(f"{'MICRO AVG':<20} {total_tp:>5} {total_fp:>5} {total_fn:>5}  "
          f"{'':>5}  {micro_precision:>10.4f} {micro_recall:>10.4f} {micro_f1:>10.4f}")

    if parse_errors:
        print("\nParse errors:")
        for e in parse_errors:
            print(f"  {e['id']}: {e['error']}")

    results = {
        "exact_match_accuracy": exact_matches / len(test_cases),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "per_slot": per_slot_metrics,
        "parse_errors": parse_errors,
        "per_sample": sample_results,
    }

    out_path = args.output or "slot_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NLU slot filling.")
    parser.add_argument("--test-file", default="test_slots.json")
    parser.add_argument("--model-name", choices=MODELS.keys(), default="qwen3")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--n-exchanges", type=int, default=10)
    parser.add_argument(
        "--price-tolerance", type=float, default=0.10,
        help="Relative tolerance for car_price comparison (default 10%%)"
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())